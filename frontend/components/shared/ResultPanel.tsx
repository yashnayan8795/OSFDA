import { AlertCircle, Loader2 } from "lucide-react";
import { ReactNode } from "react";

interface ResultPanelProps {
  isLoading?: boolean;
  error?: string | null;
  isEmpty?: boolean;
  emptyMessage?: string;
  children?: ReactNode;
}

export function ResultPanel({
  isLoading = false,
  error = null,
  isEmpty = false,
  emptyMessage = "No results to display",
  children,
}: ResultPanelProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-12">
        <Loader2 className="w-6 h-6 text-accent animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/30">
        <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
        <div className="flex flex-col gap-1">
          <p className="text-sm font-medium text-red-400">Error</p>
          <p className="text-sm text-red-300/80">{error}</p>
        </div>
      </div>
    );
  }

  if (isEmpty) {
    return (
      <div className="flex items-center justify-center p-12 text-center">
        <p className="text-muted-foreground">{emptyMessage}</p>
      </div>
    );
  }

  return <>{children}</>;
}
