"use client";

import { useState } from "react";
import { TabLayout } from "@/components/shared/TabLayout";
import { Button } from "@/components/ui/button";
import { ProbabilityBar } from "@/components/shared/ProbabilityBar";
import { ResultPanel } from "@/components/shared/ResultPanel";
import { ModelInfoCard } from "@/components/shared/ModelInfoCard";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

export default function CategoryPage() {
  const [narrative, setNarrative] = useState("");
  const [mode, setMode] = useState<"quick" | "accurate">("accurate");
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async () => {
    if (narrative.length < 50) {
      alert("Narrative must be at least 50 characters");
      return;
    }
    setIsLoading(true);
    setTimeout(() => {
      setResult({
        labels: [
          { name: "airframe", probability: 0.85, threshold: 0.5 },
          { name: "engine", probability: 0.42, threshold: 0.5 },
          { name: "fuel_system", probability: 0.28, threshold: 0.5 },
          { name: "electrical", probability: 0.15, threshold: 0.5 },
          { name: "flight_control", probability: 0.08, threshold: 0.5 },
        ],
        modelUsed: mode === "quick" ? "TF-IDF Baseline" : "Fusion",
        modelInfo: {
          name: "Category Classifier",
          version: "2.0.1",
          trainingDate: "2025-08-10",
          primaryMetric: 0.891,
          calibrationStatus: "good",
          status: "TRAINED" as const,
        },
      });
      setIsLoading(false);
    }, 1000);
  };

  return (
    <ErrorBoundary>
      <TabLayout>
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-6xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Form */}
              <div className="border border-border rounded-lg bg-card p-8">
                <div className="space-y-6">
                  <div>
                    <h2 className="text-2xl font-bold text-foreground mb-2">Category Classification</h2>
                    <p className="text-muted-foreground">Analyze incident narrative for category labels</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Incident Narrative
                    </label>
                    <textarea
                      value={narrative}
                      onChange={(e) => setNarrative(e.target.value)}
                      className="w-full px-3 py-2 bg-input border border-border rounded-md text-foreground text-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-accent"
                      placeholder="Describe the incident in detail..."
                      rows={8}
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      {narrative.length} characters (minimum 50)
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-3">
                      Analysis Mode
                    </label>
                    <div className="flex gap-3">
                      <button
                        onClick={() => setMode("quick")}
                        className={`flex-1 px-4 py-2 rounded border transition-colors ${
                          mode === "quick"
                            ? "bg-accent text-accent-foreground border-accent"
                            : "bg-muted border-border text-foreground hover:bg-muted/80"
                        }`}
                      >
                        Quick (TF-IDF)
                      </button>
                      <button
                        onClick={() => setMode("accurate")}
                        className={`flex-1 px-4 py-2 rounded border transition-colors ${
                          mode === "accurate"
                            ? "bg-accent text-accent-foreground border-accent"
                            : "bg-muted border-border text-foreground hover:bg-muted/80"
                        }`}
                      >
                        Accurate (Fusion)
                      </button>
                    </div>
                  </div>

                  <Button
                    onClick={handleSubmit}
                    disabled={isLoading || narrative.length < 50}
                    className="w-full bg-accent text-accent-foreground hover:bg-accent/90"
                  >
                    {isLoading ? "Processing..." : "Classify"}
                  </Button>
                </div>
              </div>

              {/* Result */}
              <div>
                {result ? (
                  <div className="space-y-6">
                    <div className="p-6 rounded-lg border border-border bg-card space-y-4">
                      <h3 className="text-lg font-semibold text-foreground">Category Predictions</h3>
                      {result.labels.map((label: any) => (
                        <ProbabilityBar
                          key={label.name}
                          label={label.name.replace(/_/g, " ").toUpperCase()}
                          probability={label.probability}
                          color="bg-accent"
                          predicted={label.probability > label.threshold}
                        />
                      ))}
                    </div>

                    <div className="p-4 rounded-lg border border-border bg-muted/30">
                      <p className="text-sm text-muted-foreground">
                        <span className="font-semibold text-foreground">Model:</span> {result.modelUsed}
                      </p>
                    </div>

                    <ModelInfoCard {...result.modelInfo} />
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full p-8 rounded-lg border border-dashed border-border text-center">
                    <p className="text-muted-foreground">
                      Fill out the form and submit to see results
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </TabLayout>
    </ErrorBoundary>
  );
}
