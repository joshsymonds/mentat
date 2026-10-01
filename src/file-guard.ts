import { lstatSync, realpathSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';

export interface FileGuardContext {
  memoryDir?: string;
  recordDir?: string;
  home?: string;
  cwd: string;
  transcriptPath: string;
  sessionId: string;
}

function canonicalPath(path: string): string | undefined {
  const absolutePath = isAbsolute(path) ? path : `${resolve('.')}${sep}${path}`;
  let current: string = sep;
  let missing = false;

  for (const component of absolutePath.split(sep)) {
    if (component === '' || component === '.') continue;
    if (component === '..') {
      current = dirname(current);
      continue;
    }

    const candidate = resolve(current, component);
    if (missing) {
      current = candidate;
      continue;
    }

    try {
      current = lstatSync(candidate).isSymbolicLink()
        ? realpathSync.native(candidate)
        : candidate;
    } catch (error) {
      if (
        typeof error !== 'object' ||
        error === null ||
        !('code' in error) ||
        (error.code !== 'ENOENT' && error.code !== 'ENOTDIR')
      ) {
        return undefined;
      }
      current = candidate;
      missing = true;
    }
  }
  return current;
}

function fromBase(base: string, path: string): string {
  if (isAbsolute(path)) return path;
  const absoluteBase = isAbsolute(base) ? base : resolve(base);
  return `${absoluteBase}${sep}${path}`;
}

function containsPath(root: string, target: string): boolean {
  const pathFromRoot = relative(root, target);
  return pathFromRoot === '' || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' && !isAbsolute(pathFromRoot));
}

function pathsOverlap(left: string, right: string): boolean {
  return containsPath(left, right) || containsPath(right, left);
}

export function shouldDenyFileTool(
  toolName: string,
  input: Record<string, unknown>,
  context: FileGuardContext,
): boolean {
  if (toolName !== 'Read' && toolName !== 'Glob' && toolName !== 'Grep') return false;

  const rawPath = toolName === 'Read' ? input.file_path : input.path ?? context.cwd;
  if (typeof rawPath !== 'string') return true;

  const target = canonicalPath(fromBase(context.cwd, rawPath));
  if (target === undefined) return true;

  const transcriptPath = fromBase(context.cwd, context.transcriptPath);
  const protectedRoots = [
    context.memoryDir === undefined ? undefined : fromBase(context.cwd, context.memoryDir),
    context.recordDir === undefined ? undefined : fromBase(context.cwd, context.recordDir),
    context.home === undefined ? undefined : `${fromBase(context.cwd, context.home)}${sep}.claude`,
    dirname(transcriptPath),
  ].filter((path): path is string => path !== undefined);
  const canonicalRoots = protectedRoots.map(canonicalPath);
  if (canonicalRoots.some((path) => path === undefined)) return true;

  if (toolName === 'Read') {
    const toolResults = canonicalPath(
      `${dirname(transcriptPath)}${sep}${context.sessionId}${sep}tool-results`,
    );
    if (toolResults === undefined) return true;
    if (target !== toolResults && containsPath(toolResults, target)) return false;
  }

  return canonicalRoots.some((root) => root !== undefined && pathsOverlap(root, target));
}
