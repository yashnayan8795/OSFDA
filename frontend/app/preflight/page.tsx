"use client";

import { useState } from "react";
import { TabLayout } from "@/components/shared/TabLayout";
import { DynamicForm } from "@/components/shared/DynamicForm";
import { ResultPanel } from "@/components/shared/ResultPanel";
import { ModelInfoCard } from "@/components/shared/ModelInfoCard";
import { ProbabilityBar } from "@/components/shared/ProbabilityBar";
import { Feature } from "@/types/osfda";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const mockFeatures: Feature[] = [
  {
    name: "flight_time_hours",
    type: "numeric",
    description: "Pilot flight hours",
    required: true,
  },
  {
    name: "aircraft_age_years",
    type: "numeric",
    description: "Aircraft age in years",
    required: true,
  },
  {
    name: "maintenance_interval",
    type: "numeric",
    description: "Hours since last maintenance",
    required: true,
  },
  {
    name: "weather_condition",
    type: "categorical",
    description: "Current weather",
    required: true,
    options: ["clear", "cloud", "rain", "storm", "fog"],
  },
];

export default function PreflightPage() {
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (data: Record<string, any>) => {
    setIsLoading(true);
    setTimeout(() => {
      setResult({
        riskTier: "medium",
        riskScore: 0.34,
        shaprValues: [
          { feature: "flight_hours", contribution: 0.12, direction: "positive" },
          { feature: "aircraft_age", contribution: 0.08, direction: "negative" },
          { feature: "maintenance_interval", contribution: 0.06, direction: "positive" },
        ],
        baseRateContext: {
          riskLevel: "medium",
          baseRate: 0.22,
        },
        modelInfo: {
          name: "Preflight Risk",
          version: "2.1.0",
          trainingDate: "2025-10-15",
          primaryMetric: 0.891,
          calibrationStatus: "good",
          status: "TRAINED" as const,
        },
      });
      setIsLoading(false);
    }, 1000);
  };

  const chartData = [
    { name: "flight_hours", value: 0.12 },
    { name: "aircraft_age", value: 0.08 },
    { name: "maintenance", value: 0.06 },
  ];

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
                    <h2 className="text-2xl font-bold text-foreground mb-2">Pre-Flight Risk Assessment</h2>
                    <p className="text-muted-foreground">Predict incident risk before takeoff</p>
                  </div>
                  <DynamicForm
                    features={mockFeatures}
                    onSubmit={handleSubmit}
                    isLoading={isLoading}
                    submitLabel="Assess Risk"
                  />
                </div>
              </div>

              {/* Result */}
              <div>
                {result ? (
                  <div className="space-y-6">
                    {/* Risk Tier Badge */}
                    <div className="text-center py-8 rounded-lg border border-border bg-card">
                      <p className="text-sm text-muted-foreground mb-2">Risk Tier</p>
                      <h2 className="text-4xl font-bold text-foreground capitalize">{result.riskTier}</h2>
                      <p className="text-sm text-accent font-semibold mt-2">
                        Score: {result.riskScore.toFixed(3)}
                      </p>
                    </div>

                    {/* SHAP Values */}
                    <div className="p-6 rounded-lg border border-border bg-card">
                      <h3 className="text-lg font-semibold text-foreground mb-4">Feature Contributions</h3>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={chartData}>
                          <XAxis dataKey="name" stroke="currentColor" style={{ fontSize: "12px" }} />
                          <YAxis stroke="currentColor" style={{ fontSize: "12px" }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f1117",
                              border: "1px solid #1a1d2e",
                              borderRadius: "6px",
                            }}
                          />
                          <Bar dataKey="value" fill="#06b6d4" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Base Rate Context */}
                    <div className="p-4 rounded-lg border border-border bg-muted/30">
                      <p className="text-sm text-muted-foreground">
                        <span className="font-semibold text-foreground">Base Rate ({result.baseRateContext.riskLevel}):</span>{" "}
                        {(result.baseRateContext.baseRate * 100).toFixed(1)}%
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
