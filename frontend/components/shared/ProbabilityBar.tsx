interface ProbabilityBarProps {
  label: string;
  probability: number;
  threshold?: number;
  predicted?: boolean;
  color?: string;
}

export function ProbabilityBar({
  label,
  probability,
  threshold = 0.5,
  predicted = false,
  color = "bg-accent",
}: ProbabilityBarProps) {
  const isPredicted = probability > threshold;
  const percentage = Math.round(probability * 100);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className={`text-sm font-medium ${isPredicted ? "text-foreground" : "text-muted-foreground"}`}>
          {label}
        </span>
        <span className={`text-sm font-semibold ${isPredicted ? "text-accent" : "text-muted-foreground"}`}>
          {percentage}%
        </span>
      </div>
      <div className="relative h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300 ${isPredicted ? "opacity-100" : "opacity-50"}`}
          style={{ width: `${percentage}%` }}
        />
        {threshold && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-foreground/30"
            style={{ left: `${threshold * 100}%` }}
          />
        )}
      </div>
    </div>
  );
}
