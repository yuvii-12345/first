import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LUMEN-0 — The World's Most Lightweight Video Model",
  description: "53,678 parameters • 52KB • 800ms on CPU • Generate videos from text, entirely in your browser. No GPU server required.",
  openGraph: {
    title: "LUMEN-0 — Lightweight Video Generation",
    description: "The most lightweight video generation model in the world. Runs 1000× smaller than Sora, entirely in browser.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased min-h-screen bg-[#050507] text-zinc-100">
        {children}
      </body>
    </html>
  );
}
