"use client";

import { useState } from "react";
import { TabLayout } from "@/components/shared/TabLayout";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const mockRisks = [
  { id: "1", topic: "Windscreen icing during climb", score: 0.87, growth: 2.3, severity: 0.65, count: 127, changepoint: true },
  { id: "2", topic: "Hydraulic pressure loss at altitude", score: 0.76, growth: 1.8, severity: 0.72, count: 89, changepoint: true },
  { id: "3", topic: "Navigation system intermittent failures", score: 0.64, growth: 1.2, severity: 0.45, count: 56, changepoint: false },
];

const mockTrendData = [
  { date: "Jun", score: 0.35 },
  { date: "Jul", score: 0.48 },
  { date: "Aug", score: 0.62 },
  { date: "Sep", score: 0.78 },
  { date: "Oct", score: 0.87 },
];

export default function RisksPage() {
  const [selectedRisk, setSelectedRisk] = useState(mockRisks[0]);

  return (
    <ErrorBoundary>
      <TabLayout>
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-6xl mx-auto">
            <h1 className="text-3xl font-bold text-foreground mb-8">Emerging Risks</h1>

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
                        {mockRisks.map((risk) => (
                          <tr
                            key={risk.id}
                            onClick={() => setSelectedRisk(risk)}
                            className={`hover:bg-muted/50 cursor-pointer transition-colors ${
                              selectedRisk.id === risk.id ? "bg-accent/10" : ""
                            } ${risk.changepoint ? "border-l-2 border-l-amber-500" : ""}`}
                          >
                            <td className="px-6 py-4 text-foreground font-medium">{risk.topic}</td>
                            <td className="px-6 py-4">
                              <div className="h-2 w-20 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-accent"
                                  style={{ width: `${risk.score * 100}%` }}
                                />
                              </div>
                            </td>
                            <td className="px-6 py-4 text-foreground">{risk.growth.toFixed(1)}x</td>
                            <td className="px-6 py-4 text-muted-foreground">{risk.count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Detail Panel */}
              <div className="space-y-6">
                {/* Risk Info */}
                <div className="p-6 rounded-lg border border-border bg-card">
                  <h2 className="text-xl font-bold text-foreground mb-4">{selectedRisk.topic}</h2>
                  <div className="space-y-3 text-sm">
                    <div>
                      <p className="text-muted-foreground">Risk Score</p>
                      <p className="text-lg font-semibold text-accent">{(selectedRisk.score * 100).toFixed(0)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Avg Severity</p>
                      <p className="text-lg font-semibold text-foreground">{(selectedRisk.severity * 100).toFixed(0)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Reports</p>
                      <p className="text-lg font-semibold text-foreground">{selectedRisk.count}</p>
                    </div>
                  </div>

                  {selectedRisk.changepoint && (
                    <div className="mt-4 p-3 rounded bg-amber-500/10 border border-amber-500/30 text-xs text-amber-400">
                      ⚠️ Changepoint detected in risk trajectory
                    </div>
                  )}
                </div>

                {/* Trend Chart */}
                <div className="p-6 rounded-lg border border-border bg-card">
                  <h3 className="text-sm font-semibold text-foreground mb-4">Risk Trend</h3>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={mockTrendData}>
                      <XAxis dataKey="date" stroke="currentColor" style={{ fontSize: "12px" }} />
                      <YAxis stroke="currentColor" style={{ fontSize: "12px" }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f1117",
                          border: "1px solid #1a1d2e",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#06b6d4"
                        dot={false}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>
      </TabLayout>
    </ErrorBoundary>
  );
}
