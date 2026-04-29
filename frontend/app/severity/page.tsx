"use client";

import { useState } from "react";
import { useMutation, useQuery, gql } from "@apollo/client";
import { TabLayout } from "@/components/shared/TabLayout";
import { SeverityForm } from "@/components/severity/SeverityForm";
import { SeverityResult } from "@/components/severity/SeverityResult";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";

const CLASSIFY_MUTATION = gql`
  mutation ClassifySeverity($input: SeverityInput!) {
    classifySeverity(input: $input) {
      label
      probabilities
      confidence
    }
  }
`;

const MODEL_INFO_QUERY = gql`
  query SeverityModelInfo {
    severityModelInfo {
      status
      version
      trainedAt
      primaryMetricName
      primaryMetricValue
    }
  }
`;

export default function SeverityPage() {
  const [result, setResult] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const { data: modelInfoData } = useQuery(MODEL_INFO_QUERY);
  const [classifySeverity, { loading: mutationLoading }] = useMutation(CLASSIFY_MUTATION);

  const handleSubmit = async (data: Record<string, any>) => {
    setErrorMsg(null);
    try {
      const response = await classifySeverity({ variables: { input: data } });
      const classifyData = response.data.classifySeverity;
      
      const mi = modelInfoData?.severityModelInfo || {
          version: "Unknown",
          trainedAt: "Unknown",
          primaryMetricValue: 0,
          status: "UNAVAILABLE",
      };

      setResult({
        severity: classifyData.label,
        probabilities: {
          minor: classifyData.probabilities[0] || 0,
          moderate: classifyData.probabilities[1] || 0,
          substantial: classifyData.probabilities[2] || 0,
          critical: classifyData.probabilities[3] || 0,
        },
        confidence: classifyData.confidence,
        modelInfo: {
          name: "Severity Classifier",
          version: mi.version,
          trainingDate: mi.trainedAt,
          primaryMetric: mi.primaryMetricValue,
          calibrationStatus: "N/A",
          status: mi.status,
        },
      });
    } catch (e: any) {
      console.error(e);
      setErrorMsg(e.message || "Failed to classify severity");
    }
  };

  return (
    <ErrorBoundary>
      <TabLayout>
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-6xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Form */}
              <div className="border border-border rounded-lg bg-card p-8">
                <SeverityForm onSubmit={handleSubmit} isLoading={mutationLoading} />
                {errorMsg && <p className="text-red-500 mt-4">{errorMsg}</p>}
              </div>

              {/* Result */}
              <div>
                {result ? (
                  <SeverityResult
                    {...result}
                    isLoading={mutationLoading}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full p-8 rounded-lg border border-dashed border-border text-center">
                    <p className="text-muted-foreground">
                      Fill out the form and submit to see results
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </TabLayout>
    </ErrorBoundary>
  );
}
