"use client";

import { Feature } from "@/types/osfda";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";

interface DynamicFormProps {
  features: Feature[];
  onSubmit: (data: Record<string, any>) => void;
  isLoading?: boolean;
  submitLabel?: string;
}

export function DynamicForm({
  features,
  onSubmit,
  isLoading = false,
  submitLabel = "Analyze",
}: DynamicFormProps) {
  // Build Zod schema dynamically
  const schemaObject: Record<string, z.ZodTypeAny> = {};

  features.forEach((feature) => {
    let fieldSchema: z.ZodTypeAny;

    if (feature.type === "categorical") {
      fieldSchema = z.string({ required_error: `${feature.name} is required` });
    } else if (feature.type === "numeric") {
      fieldSchema = z.coerce.number({ required_error: `${feature.name} is required` });
    } else if (feature.type === "date") {
      fieldSchema = z.string({ required_error: `${feature.name} is required` });
    } else {
      fieldSchema = z.string({ required_error: `${feature.name} is required` });
    }

    if (!feature.required) {
      fieldSchema = fieldSchema.optional();
    }

    schemaObject[feature.name] = fieldSchema;
  });

  const schema = z.object(schemaObject);
  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
  } = useForm({
    resolver: zodResolver(schema),
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {features.map((feature) => (
        <div key={feature.name} className="flex flex-col gap-2">
          <label htmlFor={feature.name} className="text-sm font-medium text-foreground">
            {feature.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
            {feature.required && <span className="text-red-400 ml-1">*</span>}
          </label>

          {feature.type === "categorical" ? (
            <Select
              onValueChange={(value) => setValue(feature.name, value)}
              required={feature.required}
            >
              <SelectTrigger id={feature.name}>
                <SelectValue placeholder={`Select ${feature.name}`} />
              </SelectTrigger>
              <SelectContent>
                {feature.options?.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : feature.type === "text" ? (
            <textarea
              {...register(feature.name)}
              id={feature.name}
              className="w-full px-3 py-2 bg-input border border-border rounded-md text-foreground text-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-accent"
              placeholder={feature.description}
              rows={6}
            />
          ) : (
            <Input
              {...register(feature.name)}
              id={feature.name}
              type={feature.type === "date" ? "date" : "number"}
              placeholder={feature.description}
              className="bg-input border-border text-foreground"
            />
          )}

          {errors[feature.name] && (
            <p className="text-xs text-red-400">
              {String(errors[feature.name]?.message || "This field is invalid")}
            </p>
          )}
        </div>
      ))}

      <Button
        type="submit"
        disabled={isLoading}
        className="w-full bg-accent text-accent-foreground hover:bg-accent/90"
      >
        {isLoading ? "Processing..." : submitLabel}
      </Button>
    </form>
  );
}
