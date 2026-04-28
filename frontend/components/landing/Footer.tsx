export function LandingFooter() {
  return (
    <footer className="py-12 px-4 sm:px-6 lg:px-8 border-t border-border bg-background">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          <div>
            <h3 className="text-sm font-semibold text-foreground mb-4">OSFDA Analytics</h3>
            <p className="text-sm text-muted-foreground">
              Integrated ML system for aviation safety incident analysis and pattern detection.
            </p>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-foreground mb-4">Technology</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>Next.js 16 + React 19</li>
              <li>Apollo Client + GraphQL CodeGen</li>
              <li>Tailwind CSS v4</li>
            </ul>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-foreground mb-4">Design</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>Dark theme with cyan/amber accents</li>
              <li>Accessible Radix UI components</li>
              <li>Type-safe GraphQL operations</li>
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-border">
          <p className="text-xs text-muted-foreground text-center">
            No frontend database. All data flows through backend GraphQL endpoint. Models versioned independently.
          </p>
        </div>
      </div>
    </footer>
  );
}
