"use client";

import { KpiTile } from "@/components/shared/KpiTile";
import { useQuery } from "@apollo/client";
import { gql } from "@apollo/client";

const MODEL_METRICS_QUERY = gql`
  query ModelMetrics {
    modelMetrics {
      id
      totalIncidents
      severityModelQwk
      emergingTopicsCount
      factorGraphNodes
    }
  }
`;

export function KpiBand() {
  const { data, loading } = useQuery(MODEL_METRICS_QUERY);

  const metrics = data?.modelMetrics || {
    totalIncidents: 38655,
    severityModelQwk: 0.742,
    emergingTopicsCount: 156,
    factorGraphNodes: 2847,
  };

  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-background via-muted/30 to-background border-y border-border">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-2xl font-bold text-foreground mb-12">System Overview</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <KpiTile
            number={metrics.totalIncidents.toLocaleString()}
            label="Total Incidents Analyzed"
            isLoading={loading}
          />
          <KpiTile
            number={`${(metrics.severityModelQwk * 100).toFixed(1)}%`}
            label="Severity Model QWK Score"
            isLoading={loading}
          />
          <KpiTile
            number={metrics.emergingTopicsCount}
            label="Emerging Risk Topics"
            isLoading={loading}
          />
          <KpiTile
            number={metrics.factorGraphNodes}
            label="Factor Graph Nodes"
            isLoading={loading}
          />
        </div>
      </div>
    </section>
  );
}
