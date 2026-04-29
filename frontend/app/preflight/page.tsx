"use client";

import { useState } from "react";
import { useMutation, useQuery, gql } from "@apollo/client";
import { TabLayout } from "@/components/shared/TabLayout";
import { DynamicForm } from "@/components/shared/DynamicForm";
import { ResultPanel } from "@/components/shared/ResultPanel";
import { ModelInfoCard } from "@/components/shared/ModelInfoCard";
import { ProbabilityBar } from "@/components/shared/ProbabilityBar";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const PREFLIGHT_SCHEMA_QUERY = gql`
  query PreflightFeatureSchema {
    preflightFeatureSchema {
      name
      type
      description
      required
      options
    }
  }
`;

const PREDICT_PREFLIGHT_MUTATION = gql`
  mutation PredictPreflightRisk($carrier: String!, $origin: String!, $destination: String!, $flightDate: String!, $depTime: String!, $aircraftType: String, $weatherForecast: String) {
    predictPreflightRisk(input: {
      carrier: $carrier,
      origin: $origin,
      destination: $destination,
      flightDate: $flightDate,
      depTime: $depTime,
      aircraftType: $aircraftType,
      weatherForecast: $weatherForecast
    }) {
      riskScore
      riskTier
      baseRateMultiple
      topContributors {
        feature
        impact
        description
      }
      modelStatus
      disclaimer
    }
  }
`;

export default function PreflightPage() {
  const [result, setResult] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const { data: schemaData, loading: schemaLoading, error: schemaError } = useQuery(PREFLIGHT_SCHEMA_QUERY);
  const [predictRisk, { loading: mutationLoading }] = useMutation(PREDICT_PREFLIGHT_MUTATION);

  const features = schemaData?.preflightFeatureSchema || [];

  const handleSubmit = async (data: Record<string, any>) => {
    setErrorMsg(null);
    try {
      const response = await predictRisk({
        variables: {
          carrier: data.carrier || "",
          origin: data.origin || "",
          destination: data.destination || "",
          flightDate: data.flight_date || data.flightDate || "",
          depTime: data.dep_time || data.depTime || "",
          aircraftType: data.aircraft_type || data.aircraftType || null,
          weatherForecast: data.weather_forecast || data.weatherForecast || null,
        },
      });
      const d = response.data.predictPreflightRisk;
      setResult({
        riskScore: d.riskScore,
        riskTier: d.riskTier,
        baseRateMultiple: d.baseRateMultiple,
        contributors: d.topContributors,
        disclaimer: d.disclaimer,
        modelStatus: d.modelStatus,
      });
    } catch (e: any) {
      console.error(e);
      setErrorMsg(e.message || "Risk prediction failed");
    }
  };

  const chartData = result?.contributors?.map((c: any) => ({
    name: c.feature,
    value: Math.abs(c.impact),
  })) || [];

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
                  {schemaError && (
                    <p className="text-red-500 text-sm">Failed to load schema. Is the backend running?</p>
                  )}
                  <DynamicForm
                    features={features}
                    onSubmit={handleSubmit}
                    isLoading={mutationLoading || schemaLoading}
                    submitLabel="Assess Risk"
                  />
                  {errorMsg && <p className="text-red-500 text-sm mt-2">{errorMsg}</p>}
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
                      <p className="text-xs text-muted-foreground mt-1">
                        {result.baseRateMultiple.toFixed(2)}x population base rate
                      </p>
                    </div>

                    {/* Feature Contributions */}
                    {chartData.length > 0 && (
                      <div className="p-6 rounded-lg border border-border bg-card">
                        <h3 className="text-lg font-semibold text-foreground mb-4">Feature Contributions</h3>
                        <ResponsiveContainer width="100%" height={200}>
                          <BarChart data={chartData}>
                            <XAxis dataKey="name" stroke="currentColor" style={{ fontSize: "12px" }} />
                            <YAxis stroke="currentColor" style={{ fontSize: "12px" }} />
                            <Tooltip
                              contentStyle={{ backgroundColor: "#0f1117", border: "1px solid #1a1d2e", borderRadius: "6px" }}
                            />
                            <Bar dataKey="value" fill="#06b6d4" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {/* Disclaimer */}
                    {result.disclaimer && (
                      <div className="p-4 rounded-lg border border-border bg-muted/30 text-xs text-muted-foreground">
                        {result.disclaimer}
                      </div>
                    )}
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
