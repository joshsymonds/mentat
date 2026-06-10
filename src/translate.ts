// Translates SDK messages into backend Events. Stateful per session: tool_use
// ids are remembered so tool results can be labeled with the tool's name (the
// wire result carries only an id). Not concurrency-safe; each session owns one.
//
// The taxonomy is deliberately exhaustive: every message type the SDK is known
// to emit is either mapped or explicitly ignored, and anything else becomes an
// `unknown` event — the signal that the SDK moved ahead of this build, which
// callers must log loudly, never drop.

import { z } from 'zod';

import type { Event, Result } from './backend.ts';

const textDeltaSchema = z.looseObject({
  type: z.literal('text_delta'),
  text: z.string(),
});

const thinkingDeltaSchema = z.looseObject({
  type: z.literal('thinking_delta'),
  thinking: z.string(),
});

const streamEventSchema = z.looseObject({
  type: z.literal('stream_event'),
  event: z.looseObject({ type: z.string() }),
});

const contentBlockDeltaSchema = z.looseObject({
  type: z.literal('content_block_delta'),
  delta: z.looseObject({ type: z.string() }),
});

/** Anthropic stream event types that carry nothing the daemon surfaces. */
const IGNORED_STREAM_EVENTS = new Set([
  'content_block_start',
  'content_block_stop',
  'message_start',
  'message_delta',
  'message_stop',
]);

/** content_block_delta variants that carry nothing the daemon surfaces. */
const IGNORED_DELTAS = new Set(['input_json_delta', 'signature_delta']);

const systemSchema = z.looseObject({
  type: z.literal('system'),
  subtype: z.string(),
});

const thinkingTokensSchema = z.looseObject({
  subtype: z.literal('thinking_tokens'),
  estimated_tokens: z.number(),
});

/** system subtypes that are session bookkeeping, not conversation events. */
const IGNORED_SYSTEM_SUBTYPES = new Set(['init', 'status']);

const toolUseBlockSchema = z.looseObject({
  type: z.literal('tool_use'),
  id: z.string(),
  name: z.string(),
});

/** assistant content block types that don't surface as events (text arrives
 * as stream deltas; thinking as thinking deltas). */
const IGNORED_ASSISTANT_BLOCKS = new Set(['text', 'thinking', 'redacted_thinking']);

const assistantSchema = z.looseObject({
  type: z.literal('assistant'),
  message: z.looseObject({
    content: z.array(z.looseObject({ type: z.string() })),
  }),
});

const toolResultBlockSchema = z.looseObject({
  type: z.literal('tool_result'),
  tool_use_id: z.string(),
  is_error: z.boolean().nullish(),
  content: z
    .union([z.string(), z.array(z.looseObject({ type: z.string(), text: z.string().optional() }))])
    .nullish(),
});

const userSchema = z.looseObject({
  type: z.literal('user'),
  message: z.looseObject({
    content: z.union([z.string(), z.array(z.looseObject({ type: z.string() }))]),
  }),
});

const resultSchema = z.looseObject({
  type: z.literal('result'),
  subtype: z.string(),
  is_error: z.boolean(),
  result: z.string().optional(),
  stop_reason: z.string().nullish(),
  session_id: z.string(),
  total_cost_usd: z.number(),
  usage: z.looseObject({
    input_tokens: z.number().optional(),
    output_tokens: z.number().optional(),
    cache_read_input_tokens: z.number().optional(),
    cache_creation_input_tokens: z.number().optional(),
  }),
});

/** Top-level SDK message types that are bookkeeping, not conversation. */
const IGNORED_MESSAGE_TYPES = new Set(['rate_limit_event']);

const messageTypeSchema = z.looseObject({ type: z.string() });

export class Translator {
  /** tool_use id → tool name, for labeling results. */
  private readonly toolNames = new Map<string, string>();

  translate(message: unknown): Event[] {
    const typed = messageTypeSchema.safeParse(message);
    if (!typed.success) {
      return [{ kind: 'unknown', raw: message }];
    }
    switch (typed.data.type) {
      case 'stream_event':
        return this.translateStreamEvent(message);
      case 'system':
        return this.translateSystem(message);
      case 'assistant':
        return this.translateAssistant(message);
      case 'user':
        return this.translateUser(message);
      case 'result':
        return this.translateResult(message);
      default:
        return IGNORED_MESSAGE_TYPES.has(typed.data.type)
          ? []
          : [{ kind: 'unknown', raw: message }];
    }
  }

  private translateStreamEvent(message: unknown): Event[] {
    const parsed = streamEventSchema.safeParse(message);
    if (!parsed.success) {
      return [{ kind: 'unknown', raw: message }];
    }
    const { event } = parsed.data;
    if (event.type === 'content_block_delta') {
      const container = contentBlockDeltaSchema.safeParse(event);
      if (!container.success) {
        return [{ kind: 'unknown', raw: message }];
      }
      const text = textDeltaSchema.safeParse(container.data.delta);
      if (text.success) {
        return [{ kind: 'textDelta', text: text.data.text }];
      }
      const thinking = thinkingDeltaSchema.safeParse(container.data.delta);
      if (thinking.success) {
        return [{ kind: 'thinkingDelta', text: thinking.data.thinking }];
      }
      return IGNORED_DELTAS.has(container.data.delta.type)
        ? []
        : [{ kind: 'unknown', raw: message }];
    }
    return IGNORED_STREAM_EVENTS.has(event.type) ? [] : [{ kind: 'unknown', raw: message }];
  }

  private translateSystem(message: unknown): Event[] {
    const parsed = systemSchema.safeParse(message);
    if (!parsed.success) {
      return [{ kind: 'unknown', raw: message }];
    }
    const thinking = thinkingTokensSchema.safeParse(parsed.data);
    if (thinking.success) {
      return [{ kind: 'thinking', tokens: thinking.data.estimated_tokens }];
    }
    return IGNORED_SYSTEM_SUBTYPES.has(parsed.data.subtype)
      ? []
      : [{ kind: 'unknown', raw: message }];
  }

  private translateAssistant(message: unknown): Event[] {
    const parsed = assistantSchema.safeParse(message);
    if (!parsed.success) {
      return [{ kind: 'unknown', raw: message }];
    }
    const events: Event[] = [];
    for (const block of parsed.data.message.content) {
      const toolUse = toolUseBlockSchema.safeParse(block);
      if (toolUse.success) {
        this.toolNames.set(toolUse.data.id, toolUse.data.name);
        events.push({ kind: 'toolStart', tool: toolUse.data.name });
      } else if (!IGNORED_ASSISTANT_BLOCKS.has(block.type)) {
        events.push({ kind: 'unknown', raw: message });
      }
    }
    return events;
  }

  private translateUser(message: unknown): Event[] {
    const parsed = userSchema.safeParse(message);
    if (!parsed.success) {
      return [{ kind: 'unknown', raw: message }];
    }
    // A plain-string user message is the echo of turn input, not an event.
    if (typeof parsed.data.message.content === 'string') {
      return [];
    }
    const events: Event[] = [];
    for (const block of parsed.data.message.content) {
      const toolResult = toolResultBlockSchema.safeParse(block);
      if (toolResult.success) {
        events.push({
          kind: 'toolResult',
          tool: this.toolNames.get(toolResult.data.tool_use_id) ?? '',
          isError: toolResult.data.is_error === true,
          content: toolResultText(toolResult.data.content),
        });
      }
      // Non-tool_result blocks (text, tool_reference, …) are conversation
      // plumbing the daemon doesn't surface.
    }
    return events;
  }

  private translateResult(message: unknown): Event[] {
    const parsed = resultSchema.safeParse(message);
    if (!parsed.success) {
      return [{ kind: 'unknown', raw: message }];
    }
    const data = parsed.data;
    const result: Result = {
      text: data.result ?? '',
      isError: data.is_error,
      stopReason: data.stop_reason ?? '',
      sessionId: data.session_id,
      costUsd: data.total_cost_usd,
      usage: {
        inputTokens: data.usage.input_tokens ?? 0,
        outputTokens: data.usage.output_tokens ?? 0,
        cacheReadInputTokens: data.usage.cache_read_input_tokens ?? 0,
        cacheCreationInputTokens: data.usage.cache_creation_input_tokens ?? 0,
      },
    };
    return [{ kind: 'done', result }];
  }
}

function toolResultText(
  content: string | { type: string; text?: string | undefined }[] | null | undefined,
): string {
  if (typeof content === 'string') {
    return content;
  }
  if (content == null) {
    return '';
  }
  return content
    .filter((block) => block.type === 'text')
    .map((block) => block.text ?? '')
    .join('');
}
