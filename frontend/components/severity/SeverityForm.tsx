"use client";

import { useQuery, gql } from "@apollo/client";
import { DynamicForm } from "@/components/shared/DynamicForm";
import { Feature } from "@/types/osfda";

const SCHEMA_QUERY = gql`
  query SeverityFeatureSchema {
    severityFeatureSchema {
      name
      type
      description
      required
      options
    }
  }
`;

interface SeverityFormProps {
  onSubmit: (data: Record<string, any>) => void;
  isLoading?: boolean;
}

export function SeverityForm({ onSubmit, isLoading }: SeverityFormProps) {
  const { data, loading, error } = useQuery(SCHEMA_QUERY);

  if (error) {
    return (
      <div className="text-red-500">
        Failed to load feature schema. Please ensure the backend is running.
      </div>
    );
  }

  const features = data?.severityFeatureSchema || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">Incident Classification</h2>
        <p className="text-muted-foreground">Enter incident details for severity assessment</p>
      </div>
      <DynamicForm
        features={features}
        onSubmit={onSubmit}
        isLoading={isLoading || loading}
        submitLabel="Classify Incident"
      />
    </div>
  );
}
