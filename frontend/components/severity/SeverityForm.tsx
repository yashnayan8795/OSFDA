"use client";

import { DynamicForm } from "@/components/shared/DynamicForm";
import { Feature } from "@/types/osfda";

interface SeverityFormProps {
  onSubmit: (data: Record<string, any>) => void;
  isLoading?: boolean;
}

// Mock features for now - will be fetched from GraphQL
const mockFeatures: Feature[] = [
  {
    name: "aircraft_type",
    type: "categorical",
    description: "Type of aircraft involved",
    required: true,
    options: ["B737", "A320", "DHC6", "CRJ200", "E175"],
  },
  {
    name: "phase_of_flight",
    type: "categorical",
    description: "Flight phase during incident",
    required: true,
    options: ["preflight", "takeoff", "cruise", "descent", "landing", "ground"],
  },
  {
    name: "injuries",
    type: "numeric",
    description: "Number of injuries",
    required: true,
  },
  {
    name: "damage_extent",
    type: "categorical",
    description: "Aircraft damage extent",
    required: true,
    options: ["none", "minor", "substantial", "destroyed"],
  },
];

export function SeverityForm({ onSubmit, isLoading }: SeverityFormProps) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">Incident Classification</h2>
        <p className="text-muted-foreground">Enter incident details for severity assessment</p>
      </div>
      <DynamicForm
        features={mockFeatures}
        onSubmit={onSubmit}
        isLoading={isLoading}
        submitLabel="Classify Incident"
      />
    </div>
  );
}
