"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { useEffect, useRef } from "react";

export function Hero() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Subtle parallax on scroll
    const handleScroll = () => {
      if (containerRef.current) {
        const offset = window.scrollY * 0.5;
        containerRef.current.style.transform = `translateY(${offset}px)`;
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-background pt-20 pb-12 px-4 sm:px-6 lg:px-8">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-accent/10 via-background to-background pointer-events-none" />

      {/* Animated grid lines */}
      <div
        ref={containerRef}
        className="absolute inset-0 overflow-hidden pointer-events-none"
      >
        <div className="absolute inset-0 opacity-20">
          {Array.from({ length: 30 }).map((_, i) => (
            <div
              key={`line-${i}`}
              className="absolute bg-accent/20"
              style={{
                width: "1px",
                height: "100%",
                left: `${(i / 30) * 100}%`,
              }}
            />
          ))}
        </div>
      </div>

      <div className="relative z-10 max-w-4xl mx-auto text-center">
        {/* Main headline with offset effect */}
        <div className="mb-8">
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-text-balance mb-4">
            <span className="text-foreground">Aviation Safety</span>
            <br />
            <span className="text-accent font-light opacity-40">Aviation Safety</span>
          </h1>
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto text-balance">
            Integrated machine learning system for incident severity, categorization, pre-flight risk assessment, emerging risk detection, and factor graph analysis.
          </p>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mt-12">
          <Link
            href="/severity"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-accent text-accent-foreground font-semibold rounded-lg hover:bg-accent/90 transition-colors"
          >
            Open Dashboard
            <ArrowRight size={18} />
          </Link>
          <a
            href="#architecture"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 border border-border text-foreground font-semibold rounded-lg hover:bg-muted transition-colors"
          >
            View Architecture
          </a>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-pulse">
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <span className="text-xs uppercase tracking-widest">Scroll</span>
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>
      </div>
    </section>
  );
}
