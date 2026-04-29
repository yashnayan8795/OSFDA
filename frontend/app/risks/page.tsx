"use client";

import { useState, useEffect } from "react";
import { useQuery, gql } from "@apollo/client";
import { TabLayout } from "@/components/shared/TabLayout";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const EMERGING_RISKS_QUERY = gql`
  query EmergingRisks($limit: Int) {
    emergingRisks(limit: $limit) {
      topicId
      name
      riskScore
      growthRatio
      recentChangepoint
      avgSeverity
      count
    }
  }
`;

const RISK_TREND_QUERY = gql`
  query RiskTrend($topicId: Int!) {
    riskTrend(topicId: $topicId) {
      period
      count
      avgSeverity
    }
  }
`;

export default function RisksPage() {
  const [selectedTopicId, setSelectedTopicId] = useState<number | null>(null);

  const { data: risksData, loading: risksLoading, error: risksError } = useQuery(EMERGING_RISKS_QUERY, {
    variables: { limit: 20 },
  });

  useEffect(() => {
    if (risksData?.emergingRisks?.length > 0 && selectedTopicId === null) {
      setSelectedTopicId(risksData.emergingRisks[0].topicId);
    }
  }, [risksData, selectedTopicId]);

  const { data: trendData } = useQuery(RISK_TREND_QUERY, {
    variables: { topicId: selectedTopicId },
    skip: selectedTopicId === null,
  });

  const risks = risksData?.emergingRisks || [];
  const selectedRisk = risks.find((r: any) => r.topicId === selectedTopicId) || risks[0];
  const trendPoints = trendData?.riskTrend || [];

  return (
    <ErrorBoundary>
      <TabLayout>
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-6xl mx-auto">
            <h1 className="text-3xl font-bold text-foreground mb-8">Emerging Risks</h1>

            {risksError && (
              <p className="text-red-500 mb-4">Failed to load risks. Is the backend running?</p>
            )}

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Risk Table */}
              <div className="lg:col-span-2">
                <div className="border border-border rounded-lg bg-card overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="border-b border-border bg-muted/50">
                        <tr>
                          <th className="px-6 py-3 text-left text-foreground font-semibold">Topic</th>
                          <th className="px-6 py-3 text-left text-foreground font-semibold">Score</th>
                          <th className="px-6 py-3 text-left text-foreground font-semibold">Growth</th>
                          <th className="px-6 py-3 text-left text-foreground font-semibold">Count</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {risksLoading
                          ? Array.from({ length: 5 }).map((_, i) => (
                              <tr key={i}>
                                <td colSpan={4} className="px-6 py-4 text-muted-foreground text-center">
                                  Loading...
                                </td>
                              </tr>
                            ))
                          : risks.map((risk: any) => (
                              <tr
                                key={risk.topicId}
                                onClick={() => setSelectedTopicId(risk.topicId)}
                                className={`hover:bg-muted/50 cursor-pointer transition-colors ${
                                  selectedTopicId === risk.topicId ? "bg-accent/10" : ""
                                } ${risk.recentChangepoint ? "border-l-2 border-l-amber-500" : ""}`}
                              >
                                <td className="px-6 py-4 text-foreground font-medium">{risk.name}</td>
                                <td className="px-6 py-4">
                                  <div className="h-2 w-20 bg-muted rounded-full overflow-hidden">
                                    <div
                                      className="h-full bg-accent"
                                      style={{ width: `${risk.riskScore * 100}%` }}
                                    />
                                  </div>
                                </td>
                                <td className="px-6 py-4 text-foreground">{risk.growthRatio.toFixed(1)}x</td>
                                <td className="px-6 py-4 text-muted-foreground">{risk.count}</td>
                              </tr>
                            ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Detail Panel */}
              {selectedRisk && (
                <div className="space-y-6">
                  <div className="p-6 rounded-lg border border-border bg-card">
                    <h2 className="text-xl font-bold text-foreground mb-4">{selectedRisk.name}</h2>
                    <div className="space-y-3 text-sm">
                      <div>
                        <p className="text-muted-foreground">Risk Score</p>
                        <p className="text-lg font-semibold text-accent">
                          {(selectedRisk.riskScore * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Avg Severity</p>
                        <p className="text-lg font-semibold text-foreground">
                          {(selectedRisk.avgSeverity * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Reports</p>
                        <p className="text-lg font-semibold text-foreground">{selectedRisk.count}</p>
                      </div>
                    </div>

                    {selectedRisk.recentChangepoint && (
                      <div className="mt-4 p-3 rounded bg-amber-500/10 border border-amber-500/30 text-xs text-amber-400">
                        ⚠️ Changepoint detected in risk trajectory
                      </div>
                    )}
                  </div>

                  {trendPoints.length > 0 && (
                    <div className="p-6 rounded-lg border border-border bg-card">
                      <h3 className="text-sm font-semibold text-foreground mb-4">Risk Trend</h3>
                      <ResponsiveContainer width="100%" height={150}>
                        <LineChart data={trendPoints}>
                          <XAxis dataKey="period" stroke="currentColor" style={{ fontSize: "12px" }} />
                          <YAxis stroke="currentColor" style={{ fontSize: "12px" }} />
                          <Tooltip contentStyle={{ backgroundColor: "#0f1117", border: "1px solid #1a1d2e" }} />
                          <Line
                            type="monotone"
                            dataKey="count"
                            stroke="#06b6d4"
                            dot={false}
                            strokeWidth={2}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </TabLayout>
    </ErrorBoundary>
  );
}
