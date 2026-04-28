import { format } from "date-fns";
import { ModelStatusBadge } from "./ModelStatusBadge";
import { ModelStatus } from "@/types/osfda";

interface ModelInfoCardProps {
  name: string;
  version: string;
  trainingDate: string;
  primaryMetric: number;
  calibrationStatus: string;
  status: ModelStatus;
}

export function ModelInfoCard({
  name,
  version,
  trainingDate,
  primaryMetric,
  calibrationStatus,
  status,
}: ModelInfoCardProps) {
  return (
    <div className="flex items-center justify-between gap-4 p-4 rounded-lg bg-muted/30 border border-border">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-foreground">{name}</h3>
          <span className="text-xs text-muted-foreground">v{version}</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>Trained: {format(new Date(trainingDate), "MMM dd, yyyy")}</span>
          <span>•</span>
          <span>Metric: {(primaryMetric * 100).toFixed(1)}%</span>
          <span>•</span>
          <span>{calibrationStatus}</span>
        </div>
      </div>
      <ModelStatusBadge status={status} />
    </div>
  );
}
