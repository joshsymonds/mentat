export type FactSource = 'josh' | 'inferred' | 'third-party';

export interface MemoryHeader {
  name: string;
  relation?: string;
  aliases: string[];
  contact?: string;
  group: string;
  summary: string;
}

const FACT_SOURCES = new Set<FactSource>(['josh', 'inferred', 'third-party']);
const HEADER_KEYS = new Set(['name', 'relation', 'aliases', 'contact', 'group', 'summary']);
const DATED_FACT = /^\[(\d{4}-\d{2}-\d{2}), (josh|inferred|third-party)\] (.+)$/;

function invalid(message: string): never {
  throw new Error(`invalid memory record: ${message}`);
}

function validSource(source: string): source is FactSource {
  return FACT_SOURCES.has(source as FactSource);
}

function validDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;

  const year = Number(value.slice(0, 4));
  const month = Number(value.slice(5, 7));
  const day = Number(value.slice(8, 10));
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const daysInMonth = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const days = daysInMonth[month - 1];
  return days !== undefined && day >= 1 && day <= days;
}

function validateHeader(header: MemoryHeader): void {
  for (const key of ['name', 'group', 'summary'] as const) {
    const value = header[key];
    if (typeof value !== 'string' || value.trim() === '') invalid(`${key} is required`);
    if (/[\r\n]/.test(value)) invalid(`${key} must be single-line`);
  }
  for (const key of ['relation', 'contact'] as const) {
    const value = header[key];
    if (value !== undefined && (typeof value !== 'string' || /[\r\n]/.test(value))) {
      invalid(`${key} must be single-line`);
    }
  }
  if (!Array.isArray(header.aliases)) invalid('aliases must be an array');
  for (const alias of header.aliases) {
    if (typeof alias !== 'string' || alias.trim() === '' || /[\r\n,]/.test(alias)) {
      invalid('aliases must be nonempty, single-line, comma-free values');
    }
  }
}

function requiredHeaderValue(values: Map<string, string>, key: string): string {
  const value = values.get(key);
  if (value === undefined || value.trim() === '') invalid(`${key} is required`);
  return value;
}

function parseFact(fact: string): string {
  if (fact === '' || fact.startsWith('- ') || /[\r\n]/.test(fact)) {
    invalid('facts must be nonempty single lines without a list marker');
  }
  if (fact.startsWith('[')) {
    const match = DATED_FACT.exec(fact);
    if (!match || !validDate(fact.slice(1, 11))) {
      invalid('dated fact must use [YYYY-MM-DD, source] text');
    }
  }
  return fact;
}

export function parseMemoryRecord(text: string): { header: MemoryHeader; facts: string[] } {
  if (typeof text !== 'string') invalid('record must be text');
  const lines = text.split('\n');
  if (lines.at(-1) === '') lines.pop();
  if (lines[0] !== '---') invalid('record must start with a header delimiter');

  const endHeader = lines.indexOf('---', 1);
  if (endHeader < 0) invalid('header must end with a delimiter');
  const values = new Map<string, string>();
  for (const line of lines.slice(1, endHeader)) {
    const match = /^([a-z]+): (.*)$/.exec(line);
    if (!match) invalid('header lines must be key: value pairs');
    const key = match[1] ?? invalid('header lines must be key: value pairs');
    if (!HEADER_KEYS.has(key)) invalid(`unknown header field ${key}`);
    if (values.has(key)) invalid(`duplicate header field ${key}`);
    values.set(key, match[2] ?? '');
  }

  const name = requiredHeaderValue(values, 'name');
  const group = requiredHeaderValue(values, 'group');
  const summary = requiredHeaderValue(values, 'summary');
  const aliasesValue = values.get('aliases') ?? '';
  const aliases = aliasesValue === '' ? [] : aliasesValue.split(',').map((alias) => alias.trim());
  if (aliases.some((alias) => alias === '')) invalid('aliases must be comma-free values');

  const header: MemoryHeader = { name, aliases, group, summary };
  const relation = values.get('relation');
  const contact = values.get('contact');
  if (relation !== undefined) header.relation = relation;
  if (contact !== undefined) header.contact = contact;
  validateHeader(header);

  const facts = lines.slice(endHeader + 1).map((line) => {
    if (!line.startsWith('- ')) invalid('facts must start with \'- \'');
    return parseFact(line.slice(2));
  });
  return { header, facts };
}

export function formatMemoryRecord(
  header: MemoryHeader,
  facts: string[],
  source: FactSource,
  now: Date,
): string {
  validateHeader(header);
  if (!validSource(source)) invalid('source is invalid');
  if (!Array.isArray(facts)) invalid('facts must be an array');
  if (!(now instanceof Date) || !Number.isFinite(now.getTime())) invalid('date must be valid');

  const fields: [string, string][] = [['name', header.name]];
  if (header.relation !== undefined) fields.push(['relation', header.relation]);
  fields.push(['aliases', header.aliases.join(', ')]);
  if (header.contact !== undefined) fields.push(['contact', header.contact]);
  fields.push(['group', header.group], ['summary', header.summary]);

  const headerLines = fields.map(([key, value]) => `${key}: ${value}`);
  const date = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
  ].join('-');
  const factLines = facts.map((fact) => {
    const parsed = parseFact(fact);
    return `- ${DATED_FACT.test(parsed) ? parsed : `[${date}, ${source}] ${parsed}`}`;
  });
  return `---\n${headerLines.join('\n')}\n---\n${factLines.length > 0 ? `${factLines.join('\n')}\n` : ''}`;
}
