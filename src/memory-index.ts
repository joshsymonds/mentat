export function formatMemoryIndex(
  records: readonly {
    id: string;
    header: { name: string; relation?: string; aliases: string[]; group: string; summary: string };
  }[],
): string {
  const groups = new Map<string, (typeof records)[number][]>();
  for (const record of records) {
    const groupRecords = groups.get(record.header.group) ?? [];
    groupRecords.push(record);
    groups.set(record.header.group, groupRecords);
  }

  const sections = [...groups.entries()]
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([group, groupRecords]) => {
      const lines = [...groupRecords]
        .sort((left, right) => {
          const nameOrder = left.header.name < right.header.name ? -1 :
            left.header.name > right.header.name ? 1 : 0;
          if (nameOrder !== 0) return nameOrder;
          return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
        })
        .map(({ id, header }) => {
          const details = [
            header.relation,
            header.aliases.length > 0
              ? `also ${header.aliases.map((alias) => `"${alias}"`).join(', ')}`
              : undefined,
          ].filter((part): part is string => Boolean(part));
          const identity = details.length > 0 ? ` (${details.join('; ')})` : '';
          return `- ${header.name}${identity} — ${header.summary} [${id}]`;
        });
      return [`## ${group}`, ...lines].join('\n');
    });

  return sections.length > 0 ? `${sections.join('\n\n')}\n` : '';
}
