import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Rearchitecture Phase 3 guide §7: "the build output contains only the app
// assets" — no source maps in the shipped build (they can leak comments/paths
// into a bundle that is meant to be a narrow, reviewed allowlist), no
// baked-in tokens, no market data. The dev server proxies /api to the
// mock/real read API on the same origin so the client never needs a
// cross-origin fetch or a second credential.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
});
