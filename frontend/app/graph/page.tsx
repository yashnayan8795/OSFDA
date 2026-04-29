"use client";

import { useState } from "react";
import { useQuery, gql } from "@apollo/client";
import { TabLayout } from "@/components/shared/TabLayout";
import { Slider } from "@/components/ui/slider";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const FACTOR_GRAPH_QUERY = gql`
  query FactorGraph($minWeight: Int, $maxNodes: Int) {
    factorGraph(minWeight: $minWeight, maxNodes: $maxNodes) {
      nodes {
        id
        count
        avgSeverity
        nodeType
        community
      }
      edges {
        source
        target
        weight
        avgSeverity
      }
    }
  }
`;

const FACTOR_PATTERNS_QUERY = gql`
  query FactorPatterns {
    factorPatterns {
      patternId
      factors
      support
      avgSeverity
      lift
    }
  }
`;

export default function GraphPage() {
  const [minWeight, setMinWeight] = useState(5);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const { data: graphData, loading: graphLoading } = useQuery(FACTOR_GRAPH_QUERY, {
    variables: { minWeight, maxNodes: 200 },
  });

  const { data: patternData } = useQuery(FACTOR_PATTERNS_QUERY);

  const nodes = graphData?.factorGraph?.nodes || [];
  const edges = graphData?.factorGraph?.edges || [];
  const patterns = patternData?.factorPatterns || [];

  const selectedNodeData = nodes.find((n: any) => n.id === selectedNode);

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
                    Minimum Edge Weight: {minWeight} &nbsp;
                    <span className="text-muted-foreground text-xs">
                      ({graphLoading ? "..." : edges.length} edges, {graphLoading ? "..." : nodes.length} nodes)
                    </span>
                  </label>
                  <Slider
                    defaultValue={[minWeight]}
                    min={1}
                    max={50}
                    step={1}
                    onValueChange={(value) => setMinWeight(value[0])}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Node List (replaces graph placeholder with real data) */}
              <div className="lg:col-span-2">
                <div className="border border-border rounded-lg bg-card overflow-hidden">
                  <div className="px-4 py-3 bg-muted/50 border-b border-border">
                    <h2 className="text-sm font-semibold text-foreground">Top Factor Nodes</h2>
                    <p className="text-xs text-muted-foreground">Click a node to explore its connections</p>
                  </div>
                  <div className="overflow-auto max-h-96">
                    {graphLoading ? (
                      <p className="px-6 py-4 text-muted-foreground text-sm">Loading graph data...</p>
                    ) : nodes.length === 0 ? (
                      <p className="px-6 py-4 text-muted-foreground text-sm">No nodes found at this weight threshold.</p>
                    ) : (
                      <table className="w-full text-sm">
                        <thead className="border-b border-border sticky top-0 bg-card">
                          <tr>
                            <th className="px-4 py-2 text-left text-foreground font-medium">Factor</th>
                            <th className="px-4 py-2 text-left text-foreground font-medium">Reports</th>
                            <th className="px-4 py-2 text-left text-foreground font-medium">Avg Severity</th>
                            <th className="px-4 py-2 text-left text-foreground font-medium">Type</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {nodes.slice(0, 50).map((node: any) => (
                            <tr
                              key={node.id}
                              onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
                              className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                                selectedNode === node.id ? "bg-accent/10" : ""
                              }`}
                            >
                              <td className="px-4 py-3 text-foreground font-medium text-xs">{node.id}</td>
                              <td className="px-4 py-3 text-muted-foreground">{node.count}</td>
                              <td className="px-4 py-3">
                                <div className="h-1.5 w-16 bg-muted rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-accent"
                                    style={{ width: `${node.avgSeverity * 100}%` }}
                                  />
                                </div>
                              </td>
                              <td className="px-4 py-3 text-xs text-muted-foreground">{node.nodeType || "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                </div>

                {/* Connections for selected node */}
                {selectedNodeData && (
                  <div className="mt-4 p-4 rounded-lg border border-border bg-card">
                    <h3 className="text-sm font-semibold text-foreground mb-3">
                      Connections for: <span className="text-accent">{selectedNodeData.id}</span>
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {edges
                        .filter((e: any) => e.source === selectedNodeData.id || e.target === selectedNodeData.id)
                        .slice(0, 20)
                        .map((e: any, i: number) => {
                          const peer = e.source === selectedNodeData.id ? e.target : e.source;
                          return (
                            <span
                              key={i}
                              className="text-xs px-2 py-1 bg-muted border border-border rounded text-foreground"
                            >
                              {peer} <span className="text-muted-foreground">({e.weight})</span>
                            </span>
                          );
                        })}
                    </div>
                  </div>
                )}
              </div>

              {/* Pattern Table */}
              <div className="space-y-4">
                <h2 className="text-lg font-semibold text-foreground">Co-Occurrence Patterns</h2>
                <div className="space-y-3">
                  {patterns.length === 0 ? (
                    <p className="text-muted-foreground text-sm">Loading patterns...</p>
                  ) : (
                    patterns.slice(0, 10).map((pattern: any) => (
                      <div
                        key={pattern.patternId}
                        className="p-4 rounded-lg border border-border bg-card hover:bg-muted/50 transition-colors"
                      >
                        <p className="text-sm font-medium text-foreground mb-2">
                          {pattern.factors.join(" → ")}
                        </p>
                        <div className="space-y-1 text-xs text-muted-foreground">
                          <div>Support: {pattern.support} incidents</div>
                          <div>Avg Severity: {(pattern.avgSeverity * 100).toFixed(0)}%</div>
                          <div>Lift: {pattern.lift.toFixed(2)}x</div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </TabLayout>
    </ErrorBoundary>
  );
}
