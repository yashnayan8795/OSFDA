import { ArrowUpRight, ArrowDownRight } from "lucide-react";

interface KpiTileProps {
  number: string | number;
  label: string;
  trend?: "up" | "down";
  change?: string;
}

export function KpiTile({ number, label, trend, change }: KpiTileProps) {
  return (
    <div className="flex flex-col gap-2 p-6 rounded-lg bg-card border border-border hover:bg-muted/50 transition-colors">
      <div className="flex items-baseline gap-2">
        <div className="text-3xl font-bold text-accent">{number}</div>
        {trend && change && (
          <div className={`flex items-center gap-1 text-sm font-medium ${
            trend === "up" ? "text-green-400" : "text-red-400"
          }`}>
            {trend === "up" ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
            <span>{change}</span>
          </div>
        )}
      </div>
      <p className="text-sm text-muted-foreground">{label}</p>
    </div>
  );
}
