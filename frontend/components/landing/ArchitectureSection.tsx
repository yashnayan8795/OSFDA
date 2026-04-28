import { Database, Workflow, Zap } from "lucide-react";

export function ArchitectureSection() {
  return (
    <section id="architecture" className="py-20 px-4 sm:px-6 lg:px-8 bg-muted/30 border-y border-border">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-foreground mb-4">System Architecture</h2>
        <p className="text-lg text-muted-foreground mb-16">
          Five specialized models unified through GraphQL, enabling rapid iteration and model versioning without frontend changes.
        </p>

        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            {[
              {
                icon: Zap,
                title: "GraphQL Unified Interface",
                description:
                  "Single /graphql endpoint abstracts model selection, versioning, and feature schema evolution. Models deploy independently.",
              },
              {
                icon: Database,
                title: "Model Store Backend",
                description:
                  "REST/gRPC services for each model (Severity, Category, Preflight, Risks, Graph). Cached metadata and base rates. Configurable base-rate blending.",
              },
              {
                icon: Workflow,
                title: "Frontend-Agnostic Design",
                description:
                  "Forms auto-generate from FeatureSchema queries. Mutations handle model state (TRAINED/STUB/UNAVAILABLE) gracefully.",
              },
            ].map((item, i) => {
              const Icon = item.icon;
              return (
                <div key={i} className="flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="flex items-center justify-center h-12 w-12 rounded-lg bg-accent/20">
                      <Icon className="h-6 w-6 text-accent" />
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground">{item.title}</h3>
                    <p className="mt-2 text-muted-foreground text-sm">{item.description}</p>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="rounded-lg border border-border bg-card p-8 font-mono text-sm">
            <div className="space-y-4 text-muted-foreground">
              <div>
                <span className="text-accent">Frontend</span>
                <span className="text-muted-foreground">: Next.js + Apollo Client</span>
              </div>
              <div className="relative h-8 flex items-center">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full h-px bg-gradient-to-r from-accent/50 via-accent/20 to-transparent" />
                </div>
                <span className="relative px-2 bg-card text-accent text-xs uppercase tracking-wide">
                  GraphQL
                </span>
              </div>
              <div>
                <span className="text-accent">Backend</span>
                <span className="text-muted-foreground">: 5 Model Services</span>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-4 pt-4 border-t border-border">
                {[
                  "Severity Classifier",
                  "Category Classifier",
                  "Preflight Risk",
                  "Risk Detection",
                ].map((model) => (
                  <div key={model} className="text-xs text-muted-foreground">
                    ✓ {model}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
