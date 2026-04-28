"use client";

import { useState } from "react";
import { TabLayout } from "@/components/shared/TabLayout";
import { SeverityForm } from "@/components/severity/SeverityForm";
import { SeverityResult } from "@/components/severity/SeverityResult";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

export default function SeverityPage() {
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (data: Record<string, any>) => {
    setIsLoading(true);
    // Mock result for now
    setTimeout(() => {
      setResult({
        severity: "substantial",
        probabilities: {
          minor: 0.05,
          moderate: 0.15,
          substantial: 0.62,
          critical: 0.18,
        },
        confidence: 0.78,
        modelInfo: {
          name: "Severity Classifier",
          version: "3.2.1",
          trainingDate: "2025-09-20",
          primaryMetric: 0.742,
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
                <SeverityForm onSubmit={handleSubmit} isLoading={isLoading} />
              </div>

              {/* Result */}
              <div>
                {result ? (
                  <SeverityResult
                    {...result}
                    isLoading={isLoading}
                  />
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
