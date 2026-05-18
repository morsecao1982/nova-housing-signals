import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/",
        destination: "/nova",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
