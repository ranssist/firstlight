import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import App from "@/App"
import { Toaster } from "@/components/ui/sonner"
import { ThemeProvider } from "@/lib/theme"
import "@/index.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
      <Toaster position="bottom-right" richColors closeButton />
    </ThemeProvider>
  </StrictMode>,
)
