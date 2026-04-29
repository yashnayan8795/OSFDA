"use client";

import { useState } from "react";
import { useMutation, gql } from "@apollo/client";
import { TabLayout } from "@/components/shared/TabLayout";
import { Button } from "@/components/ui/button";
import { ProbabilityBar } from "@/components/shared/ProbabilityBar";
import { ModelInfoCard } from "@/components/shared/ModelInfoCard";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const CLASSIFY_CATEGORIES_MUTATION = gql`
  mutation ClassifyCategories($narrative: String!, $synopsis: String, $accurate: Boolean) {
    classifyCategories(input: { narrative: $narrative, synopsis: $synopsis, accurate: $accurate }) {
      predictions {
        label
        displayName
        probability
        predicted
        thresholdUsed
      }
    }
  }
`;

export default function CategoryPage() {
  const [narrative, setNarrative] = useState("");
  const [mode, setMode] = useState<"quick" | "accurate">("accurate");
  const [result, setResult] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [classifyCategories, { loading }] = useMutation(CLASSIFY_CATEGORIES_MUTATION);

  const handleSubmit = async () => {
    if (narrative.length < 50) {
      alert("Narrative must be at least 50 characters");
      return;
    }
    setErrorMsg(null);
    try {
      const response = await classifyCategories({
        variables: { narrative, synopsis: "", accurate: mode === "accurate" },
      });
      const predictions = response.data.classifyCategories.predictions;
      setResult({
        labels: predictions.map((p: any) => ({
          name: p.displayName || p.label,
          probability: p.probability,
          threshold: p.thresholdUsed,
          predicted: p.predicted,
        })),
        modelUsed: mode === "quick" ? "TF-IDF Baseline" : "Fusion",
      });
    } catch (e: any) {
      console.error(e);
      setErrorMsg(e.message || "Classification failed");
    }
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
                    disabled={loading || narrative.length < 50}
                    className="w-full bg-accent text-accent-foreground hover:bg-accent/90"
                  >
                    {loading ? "Processing..." : "Classify"}
                  </Button>

                  {errorMsg && <p className="text-red-500 text-sm">{errorMsg}</p>}
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
                          color={label.predicted ? "bg-accent" : "bg-muted-foreground"}
                          predicted={label.predicted}
                        />
                      ))}
                    </div>

                    <div className="p-4 rounded-lg border border-border bg-muted/30">
                      <p className="text-sm text-muted-foreground">
                        <span className="font-semibold text-foreground">Model:</span> {result.modelUsed}
                      </p>
                    </div>
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
