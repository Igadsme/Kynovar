"use client";

import katex from "katex";
import { useMemo } from "react";

export function Equation({ latex, prefix = "F = " }: { latex: string; prefix?: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(`${prefix}${latex}`, { throwOnError: false, displayMode: false });
    } catch {
      return latex;
    }
  }, [latex, prefix]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
