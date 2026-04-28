import { z } from "zod";

// Model state enums
export enum ModelStatus {
  TRAINED = "TRAINED",
  STUB = "STUB",
  UNAVAILABLE = "UNAVAILABLE",
}

// GraphQL-derived types (will be re-exported from codegen)
export type ModelMetadata = {
  id: string;
  name: string;
  version: string;
  trainingDate: string;
  primaryMetric: number;
  calibrationStatus: string;
  status: ModelStatus;
};

export type Feature = {
  name: string;
  type: "categorical" | "numeric" | "date" | "text";
  description: string;
  required: boolean;
  options?: string[];
};

export type SeverityInput = {
  aircraft: string;
  environment: string;
  crew: string;
  operations: string;
};

export type CategoryInput = {
  narrative: string;
  synopsis?: string;
  mode?: "quick" | "accurate";
};

export type FlightInput = {
  flight: string;
  aircraft: string;
  weather: string;
  operational: string;
};

// Zod schemas for validation
export const SeverityInputSchema = z.object({
  aircraft: z.string().min(1),
  environment: z.string().min(1),
  crew: z.string().min(1),
  operations: z.string().min(1),
});

export const CategoryInputSchema = z.object({
  narrative: z.string().min(50, "Narrative must be at least 50 characters"),
  synopsis: z.string().optional(),
  mode: z.enum(["quick", "accurate"]).default("accurate"),
});

export const FlightInputSchema = z.object({
  flight: z.string().min(1),
  aircraft: z.string().min(1),
  weather: z.string().min(1),
  operational: z.string().min(1),
});

// Severity colors for consistency
export const SEVERITY_COLORS = {
  minor: "#10b981",
  moderate: "#f59e0b",
  substantial: "#f97316",
  critical: "#ef4444",
} as const;

// Risk tier colors
export const RISK_COLORS = {
  low: "#10b981",
  medium: "#f59e0b",
  high: "#ef4444",
} as const;
