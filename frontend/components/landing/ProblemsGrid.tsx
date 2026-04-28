import Link from "next/link";

const problems = [
  {
    id: "A",
    title: "Incident Severity",
    description: "Classify safety incidents into severity tiers (Minor, Moderate, Substantial, Critical) with cost-sensitive confidence scoring.",
    metric: "QWK: 0.742",
    href: "/severity",
  },
  {
    id: "B",
    title: "Category Classification",
    description: "Multi-label classification of incident reports into maintenance, system, and operational categories with dual-mode inference.",
    metric: "F1: 0.891",
    href: "/category",
  },
  {
    id: "C",
    title: "Pre-Flight Risk",
    description: "Real-time prediction of flight-phase incident risk before takeoff, with SHAP explainability and base-rate calibration.",
    metric: "ROC-AUC: 0.876",
    href: "/preflight",
  },
  {
    id: "D",
    title: "Emerging Risks",
    description: "Detect emerging safety hazards through topic modeling with changepoint detection and trend analysis on historical corpus.",
    metric: "156 topics tracked",
    href: "/risks",
  },
  {
    id: "E",
    title: "Factor Graph",
    description: "Visualize co-occurrence patterns of failure modes and contributing factors through an interactive force-directed network.",
    metric: "2,847 nodes",
    href: "/graph",
  },
];

export function ProblemsGrid() {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-background">
      <div className="max-w-6xl mx-auto">
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-foreground mb-4">Five Models, One System</h2>
          <p className="text-lg text-muted-foreground">
            Integrated ML capabilities for comprehensive aviation incident analysis
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {problems.map((problem) => (
            <Link
              key={problem.id}
              href={problem.href}
              className="group flex flex-col p-6 rounded-lg border border-border bg-card hover:bg-muted/50 hover:border-accent/50 transition-all duration-300"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-accent/20 flex items-center justify-center">
                  <span className="text-lg font-bold text-accent">{problem.id}</span>
                </div>
              </div>

              <h3 className="text-lg font-semibold text-foreground mb-2 group-hover:text-accent transition-colors">
                {problem.title}
              </h3>

              <p className="text-sm text-muted-foreground flex-grow mb-4">
                {problem.description}
              </p>

              <div className="flex items-center justify-between pt-4 border-t border-border">
                <span className="text-xs font-medium text-accent">{problem.metric}</span>
                <span className="text-accent group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
