"use client";

import { useState } from "react";
import { TabLayout } from "@/components/shared/TabLayout";
import { Slider } from "@/components/ui/slider";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const mockPatterns = [
  { id: "1", pattern: "Icing → Loss → Speed Reduction", support: 0.15, severity: 0.71, lift: 1.94 },
  { id: "2", pattern: "Electrical → Navigation → Routing Error", support: 0.12, severity: 0.58, lift: 1.67 },
  { id: "3", pattern: "Weather → Visibility → Collision Risk", support: 0.18, severity: 0.82, lift: 2.12 },
];

export default function GraphPage() {
  const [minWeight, setMinWeight] = useState(0.1);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  return (
    <ErrorBoundary>
      <TabLayout>
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-6xl mx-auto">
            <h1 className="text-3xl font-bold text-foreground mb-8">Factor Graph</h1>

            {/* Controls */}
            <div className="mb-8 p-6 rounded-lg border border-border bg-card">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-3">
                    Minimum Edge Weight: {minWeight.toFixed(2)}
                  </label>
                  <Slider
                    defaultValue={[minWeight]}
                    max={1}
                    step={0.05}
                    onValueChange={(value) => setMinWeight(value[0])}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Graph Placeholder */}
              <div className="lg:col-span-2">
                <div className="h-96 rounded-lg border border-border bg-muted/30 flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-muted-foreground mb-2">Interactive Force-Directed Graph</p>
                    <p className="text-xs text-muted-foreground">
                      Nodes: factors | Edges: co-occurrence weight
                    </p>
                  </div>
                </div>
              </div>

              {/* Pattern Table */}
              <div className="space-y-4">
                <h2 className="text-lg font-semibold text-foreground">Co-Occurrence Patterns</h2>
                <div className="space-y-3">
                  {mockPatterns.map((pattern) => (
                    <div
                      key={pattern.id}
                      className="p-4 rounded-lg border border-border bg-card hover:bg-muted/50 cursor-pointer transition-colors"
                      onClick={() => setSelectedNode(pattern.id)}
                    >
                      <p className="text-sm font-medium text-foreground mb-2">{pattern.pattern}</p>
                      <div className="space-y-1 text-xs text-muted-foreground">
                        <div>Support: {(pattern.support * 100).toFixed(1)}%</div>
                        <div>Avg Severity: {(pattern.severity * 100).toFixed(0)}%</div>
                        <div>Lift: {pattern.lift.toFixed(2)}x</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </TabLayout>
    </ErrorBoundary>
  );
}
