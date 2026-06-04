import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "ReTreVal — Reasoning Tree with Validation",
  description:
    "Watch an LLM grow a validated reasoning tree, refine each node with tools, and remember what works — live.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f8fb" },
    { media: "(prefers-color-scheme: dark)", color: "#07070b" },
  ],
  width: "device-width",
  initialScale: 1,
};

// Set the theme class before first paint to avoid a flash. Default: dark.
const THEME_INIT = `
(function(){try{
  if(localStorage.getItem('retreval-theme')==='light'){
    document.documentElement.classList.remove('dark');
  }else{
    document.documentElement.classList.add('dark');
  }
}catch(e){document.documentElement.classList.add('dark');}})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${inter.variable} ${mono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT }} />
      </head>
      <body data-testid="app-ready" className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
