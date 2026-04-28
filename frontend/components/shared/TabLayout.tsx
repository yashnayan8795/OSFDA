"use client";

import Link from "next/link";
import { ReactNode } from "react";
import { ModelStatusBadge } from "./ModelStatusBadge";
import { ModelStatus } from "@/types/osfda";
import { usePathname } from "next/navigation";

interface TabLayoutProps {
  children: ReactNode;
  modelStatus?: ModelStatus;
}

const tabs = [
  { label: "A: Severity", href: "/severity" },
  { label: "B: Category", href: "/category" },
  { label: "C: Preflight", href: "/preflight" },
  { label: "D: Risks", href: "/risks" },
  { label: "E: Factor Graph", href: "/graph" },
];

export function TabLayout({ children, modelStatus = ModelStatus.TRAINED }: TabLayoutProps) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur">
        <div className="px-4 py-4 sm:px-6 sm:py-5">
          <div className="flex items-center justify-between gap-4">
            <h1 className="text-xl sm:text-2xl font-bold text-accent">OSFDA Analytics</h1>
            <div className="flex items-center gap-4">
              <span className="hidden sm:inline text-xs text-muted-foreground px-3 py-1 bg-muted/50 rounded">
                {process.env.NEXT_PUBLIC_GRAPHQL_URL?.replace(/^https?:\/\//, "").split("/")[0] || "local"}
              </span>
              <ModelStatusBadge status={modelStatus} />
            </div>
          </div>

          {/* Tabs */}
          <nav className="flex gap-2 mt-4 -mx-4 px-4 overflow-x-auto">
            {tabs.map((tab) => {
              const isActive = pathname === tab.href || pathname.startsWith(tab.href + "/");
              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  className={`whitespace-nowrap px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                    isActive
                      ? "border-accent text-accent"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tab.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1">{children}</main>
    </div>
  );
}
