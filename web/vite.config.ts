import path from "node:path"

import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { viteSingleFile } from "vite-plugin-singlefile"

/** 두 가지 산출물을 만든다.
 *
 *   기본        — FastAPI 가 서빙하는 실시간 관제 화면 (web/dist)
 *   --mode demo — 백엔드 없이 도는 자립형 단일 HTML (web/dist-demo)
 *
 * 데모는 공유용이다. 아티팩트/정적 호스팅에는 서버가 없고 CSP 가 외부
 * 요청을 막으므로, JS·CSS·폰트·지도배경을 전부 한 파일에 인라인한다.
 */
export default defineConfig(({ mode }) => {
  const isDemo = mode === "demo"

  return {
    plugins: [react(), tailwindcss(), ...(isDemo ? [viteSingleFile()] : [])],
    resolve: {
      // import.meta.dirname — __dirname 은 Vite 의 네이티브 설정 로더에서 안 된다.
      alias: { "@": path.resolve(import.meta.dirname, "./src") },
    },
    define: {
      "import.meta.env.VITE_DEMO": JSON.stringify(isDemo ? "1" : "0"),
    },
    build: {
      outDir: isDemo ? "dist-demo" : "dist",
      emptyOutDir: true,
      // 단일 파일로 합칠 것이므로 자산 인라인 한도를 올린다.
      assetsInlineLimit: isDemo ? 100_000_000 : 4096,
      // 폰트·음영기복도가 base64 로 들어가 청크가 커진다. 의도한 것이다.
      chunkSizeWarningLimit: isDemo ? 10_000 : 500,
    },
    server: {
      port: 5173,
      // 개발 중에는 Vite 가 프런트를, FastAPI 가 API 를 담당한다.
      // 이렇게 두면 HMR 을 쓰면서도 실제 이벤트 DB 를 그대로 본다.
      proxy: {
        "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      },
    },
  }
})
