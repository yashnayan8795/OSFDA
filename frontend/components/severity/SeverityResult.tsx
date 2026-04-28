"use client";

import { ProbabilityBar } from "@/components/shared/ProbabilityBar";
import { ResultPanel } from "@/components/shared/ResultPanel";
import { ModelInfoCard } from "@/components/shared/ModelInfoCard";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import { SEVERITY_COLORS } from "@/types/osfda";

interface SeverityResultProps {
  severity: string;
  probabilities: {
    minor: number;
    moderate: number;
    substantial: number;
    critical: number;
  };
  confidence: number;
  modelInfo: {
    name: string;
    version: string;
    trainingDate: string;
    primaryMetric: number;
    calibrationStatus: string;
    status: "TRAINED" | "STUB" | "UNAVAILABLE";
  };
  isLoading?: boolean;
  error?: string | null;
}

const severityMap = {
  minor: 0,
  moderate: 1,
  substantial: 2,
  critical: 3,
};

export function SeverityResult({
  severity,
  probabilities,
  confidence,
  modelInfo,
  isLoading = false,
  error = null,
}: SeverityResultProps) {
  const severityIndex = severityMap[severity as keyof typeof severityMap] || 0;
  const severityColor = Object.values(SEVERITY_COLORS)[severityIndex];

  const confidenceData = [
    { name: "Confidence", value: confidence * 100, fill: "#06b6d4" },
    { name: "Uncertainty", value: (1 - confidence) * 100, fill: "#1a1d2e" },
  ];

  return (
    <ResultPanel isLoading={isLoading} error={error}>
      <div className="space-y-8">
        {/* Severity Badge */}
        <div className="text-center py-8 rounded-lg border border-border bg-card">
          <p className="text-sm text-muted-foreground mb-2">Predicted Severity</p>
          <div className="flex items-baseline justify-center gap-3">
            <div
              className="w-6 h-6 rounded-full"
              style={{ backgroundColor: severityColor }}
            />
            <h2 className="text-4xl font-bold text-foreground capitalize">{severity}</h2>
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
        </div>

        {/* Probability Bars */}
        <div className="space-y-4 p-6 rounded-lg border border-border bg-card">
          <h3 className="text-lg font-semibold text-foreground">Class Probabilities</h3>
          <div className="space-y-4">
            <ProbabilityBar
              label="Minor"
              probability={probabilities.minor}
              color="bg-green-500"
              predicted={severity === "minor"}
            />
            <ProbabilityBar
              label="Moderate"
              probability={probabilities.moderate}
              color="bg-amber-500"
              predicted={severity === "moderate"}
            />
            <ProbabilityBar
              label="Substantial"
              probability={probabilities.substantial}
              color="bg-orange-500"
              predicted={severity === "substantial"}
            />
            <ProbabilityBar
              label="Critical"
              probability={probabilities.critical}
              color="bg-red-500"
              predicted={severity === "critical"}
            />
          </div>
        </div>

        {/* Confidence Ring */}
        <div className="p-6 rounded-lg border border-border bg-card">
          <h3 className="text-lg font-semibold text-foreground mb-6">Model Confidence</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={confidenceData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={2}
                dataKey="value"
              >
                {confidenceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <p className="text-center text-sm text-muted-foreground mt-4">
            Model is {(confidence * 100).toFixed(1)}% confident in this prediction
          </p>
        </div>

        {/* Model Info Footer */}
        <ModelInfoCard
          name={modelInfo.name}
          version={modelInfo.version}
          trainingDate={modelInfo.trainingDate}
          primaryMetric={modelInfo.primaryMetric}
          calibrationStatus={modelInfo.calibrationStatus}
          status={modelInfo.status}
        />
      </div>
    </ResultPanel>
  );
}
