/** 테마 — shadcn 은 `.dark` 클래스로 다크 모드를 켠다.
 *
 * 기본값은 "system" 이다. 관제실은 어두운 경우가 많지만 낮에 노트북으로
 * 여는 사람도 있어서, OS 설정을 따르되 수동 전환을 열어 둔다.
 */

import { createContext, useContext, useEffect, useState } from "react"

type Theme = "dark" | "light" | "system"

const STORAGE_KEY = "firstlight-theme"

const ThemeContext = createContext<{
  theme: Theme
  resolved: "dark" | "light"
  setTheme: (theme: Theme) => void
}>({ theme: "system", resolved: "light", setTheme: () => {} })

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(
    () => (localStorage.getItem(STORAGE_KEY) as Theme) || "system",
  )
  const [resolved, setResolved] = useState<"dark" | "light">("light")

  useEffect(() => {
    const root = document.documentElement
    const media = window.matchMedia("(prefers-color-scheme: dark)")

    const apply = () => {
      const next = theme === "system" ? (media.matches ? "dark" : "light") : theme
      root.classList.toggle("dark", next === "dark")
      root.style.colorScheme = next
      setResolved(next)
    }

    apply()
    // "system" 일 때만 OS 변경을 따라간다.
    if (theme === "system") {
      media.addEventListener("change", apply)
      return () => media.removeEventListener("change", apply)
    }
  }, [theme])

  const setTheme = (next: Theme) => {
    localStorage.setItem(STORAGE_KEY, next)
    setThemeState(next)
  }

  return (
    <ThemeContext.Provider value={{ theme, resolved, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export const useTheme = () => useContext(ThemeContext)
