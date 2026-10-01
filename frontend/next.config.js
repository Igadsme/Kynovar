/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  transpilePackages: ["three"],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.KYNOVAR_API_INTERNAL ?? "http://127.0.0.1:8000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
