import { ModelStatus } from "@/types/osfda";

interface ModelStatusBadgeProps {
  status: ModelStatus;
}

const statusConfig = {
  [ModelStatus.TRAINED]: {
    label: "Trained",
    color: "bg-green-500/20 text-green-400 border-green-500/40",
  },
  [ModelStatus.STUB]: {
    label: "Stub Mode",
    color: "bg-amber-500/20 text-amber-400 border-amber-500/40",
  },
  [ModelStatus.UNAVAILABLE]: {
    label: "Unavailable",
    color: "bg-red-500/20 text-red-400 border-red-500/40",
  },
};

export function ModelStatusBadge({ status }: ModelStatusBadgeProps) {
  const config = statusConfig[status];
  return (
    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium border ${config.color}`}>
      {config.label}
    </span>
  );
}
