/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The UI is a pure client-side SPA (the agent lives behind the FastAPI
  // bridge), so we export fully static assets. Serve `out/` with any static
  // file server — no Node runtime required in production.
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
