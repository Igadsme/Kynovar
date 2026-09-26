import "katex/dist/katex.min.css";
import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Kynovar — Autonomous Scientific Discovery",
  description: "An autonomous scientist discovering the hidden laws of a simulated universe.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
