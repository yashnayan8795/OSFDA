# OSFDA Analytics — Aviation Safety Intelligence

A research-grade, integrated ML system for analyzing aviation incidents through five specialized models: Severity Classification, Category Classification, Pre-Flight Risk Assessment, Emerging Risk Detection, and Factor Graph Analysis.

## Features

- **Tab A: Incident Severity** — Classify incidents into severity tiers (Minor, Moderate, Substantial, Critical) with cost-sensitive confidence scoring
- **Tab B: Category Classification** — Multi-label classification of incident reports with dual-mode inference (Quick TF-IDF or Accurate Fusion)
- **Tab C: Pre-Flight Risk** — Real-time incident risk prediction before takeoff with SHAP explainability and base-rate calibration
- **Tab D: Emerging Risks** — Detect emerging safety hazards with topic modeling, changepoint detection, and trend analysis
- **Tab E: Factor Graph** — Interactive visualization of co-occurrence patterns and failure mode relationships

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript
- **State & Data**: Apollo Client 4, GraphQL CodeGen
- **Styling**: Tailwind CSS v4, dark theme with cyan/amber accents
- **UI Components**: Radix UI, shadcn/ui
- **Visualizations**: Recharts, react-force-graph-2d
- **Forms**: react-hook-form, Zod validation

## Project Structure

```
.
├── app/
│   ├── layout.tsx              # Root layout with Apollo provider
│   ├── page.tsx                # Landing page
│   ├── severity/               # Tab A: Severity classification
│   ├── category/               # Tab B: Category classification
│   ├── preflight/              # Tab C: Pre-flight risk
│   ├── risks/                  # Tab D: Emerging risks
│   └── graph/                  # Tab E: Factor graph
├── components/
│   ├── landing/                # Landing page sections
│   ├── shared/                 # Shared components
│   │   ├── DynamicForm.tsx     # Auto-generated forms from GraphQL schema
│   │   ├── TabLayout.tsx       # Tab navigation
│   │   ├── ModelStatusBadge.tsx
│   │   ├── KpiTile.tsx
│   │   ├── ProbabilityBar.tsx
│   │   └── ...
│   ├── severity/               # Tab A components
│   └── ui/                     # Base UI components
├── lib/
│   ├── apollo/
│   │   ├── client.ts           # Apollo client setup
│   │   ├── provider.tsx        # Apollo provider wrapper
│   │   └── mocks.ts            # Mock resolvers for offline dev
│   ├── graphql/                # .graphql operation files
│   │   ├── shared.graphql
│   │   ├── landing.graphql
│   │   ├── severity.graphql
│   │   ├── category.graphql
│   │   ├── preflight.graphql
│   │   ├── risks.graphql
│   │   └── graph.graphql
│   ├── formatters.ts           # Format utilities
│   ├── colors.ts               # Color mapping functions
│   └── utils.ts
├── types/
│   └── osfda.ts                # OSFDA-specific types and Zod schemas
├── codegen.yml                 # GraphQL CodeGen config
└── tailwind.config.ts          # Dark theme configuration
```

## Setup Instructions

### 1. Install Dependencies

```bash
pnpm install
```

### 2. Configure Environment

Copy `.env.example` to `.env.local` and configure:

```env
NEXT_PUBLIC_GRAPHQL_URL=http://localhost:8000/graphql
NEXT_PUBLIC_USE_MOCKS=false
```

For offline development with mock data:

```env
NEXT_PUBLIC_USE_MOCKS=true
```

### 3. Generate GraphQL Types

```bash
pnpm exec graphql-codegen
```

This generates `lib/graphql/generated.ts` with type-safe React hooks for all queries and mutations.

### 4. Start Development Server

```bash
pnpm dev
```

Navigate to `http://localhost:3000` and explore the landing page or jump directly to any tab.

## GraphQL Integration

All data flows through a single `/graphql` endpoint. The frontend uses:

- **Queries**: Fetch model metadata, feature schemas, distributions, and analysis results
- **Mutations**: Trigger predictions and classifications
- **Model State Handling**: Gracefully handle TRAINED/STUB/UNAVAILABLE model states

### Example Query

```graphql
query SeverityFeatureSchema {
  severityFeatureSchema {
    name
    type
    description
    required
    options
  }
}
```

### Example Mutation

```graphql
mutation ClassifySeverity($input: SeverityInput!) {
  classifySeverity(input: $input) {
    severity
    probabilities {
      minor
      moderate
      substantial
      critical
    }
    confidence
    modelInfo {
      version
      trainingDate
      calibrationStatus
    }
  }
}
```

## Backend Integration

To connect to your backend:

1. Implement the GraphQL schema defined in `lib/graphql/*.graphql`
2. Update `NEXT_PUBLIC_GRAPHQL_URL` to point to your backend endpoint
3. Ensure models implement the correct request/response structures
4. The frontend will automatically:
   - Generate form fields from `FeatureSchema` queries
   - Display results with proper error handling
   - Show model state badges (TRAINED/STUB/UNAVAILABLE)

## Dark Theme Customization

The entire app uses a cohesive dark palette defined in `tailwind.config.ts`:

- **Base**: `#0a0e1a` (darkest background)
- **Surface**: `#0f1117` (cards and containers)
- **Primary Accent**: Cyan `#06b6d4`
- **Secondary Accent**: Amber `#f59e0b`
- **Semantic Colors**: Green (success), Red (danger), etc.

All color tokens are CSS variables, making it easy to theme consistently.

## Development Tips

### Working Offline with Mock Data

Enable mock mode to develop without a backend:

```env
NEXT_PUBLIC_USE_MOCKS=true
```

The `lib/apollo/mocks.ts` file contains realistic mock resolvers for all queries and mutations.

### Type Safety

All GraphQL operations are fully typed through CodeGen. To add a new query or mutation:

1. Add `.graphql` file in `lib/graphql/`
2. Run `pnpm exec graphql-codegen`
3. Import auto-generated hooks in components

### Form Auto-Generation

The `DynamicForm` component reads `FeatureSchema` queries and automatically renders inputs by field type (categorical, numeric, date, text). No hardcoded form fields needed.

### Adding a New Tab

1. Create `app/[tab-name]/page.tsx` with `<TabLayout>` wrapper
2. Build components in `components/[tab-name]/`
3. Add GraphQL queries in `lib/graphql/[tab-name].graphql`
4. Update tab navigation in `TabLayout` component

## Error Handling

- **GraphQL Errors**: Displayed inline with retry options
- **Network Errors**: Graceful fallbacks, cache-first strategy
- **Model Errors**: Model state badges show STUB or UNAVAILABLE status
- **Form Validation**: Zod schemas on client, Apollo validation on server

## Performance

- **Cache Strategy**: Cache-and-network for most queries, network-only for mutations
- **Code Splitting**: Route-based, automatic with Next.js
- **Image Optimization**: Next.js Image component for all assets
- **Bundle Size**: Optimized with tree-shaking and dynamic imports

## Deployment

Deploy to Vercel with one command:

```bash
vercel deploy
```

Environment variables are configured in Vercel project settings. Ensure `NEXT_PUBLIC_GRAPHQL_URL` points to your production backend.

## Testing & Validation

- **Type Safety**: 100% TypeScript coverage via GraphQL CodeGen
- **Form Validation**: Zod schemas for all inputs
- **Error Boundaries**: React error boundaries on all tabs
- **Mock Mode**: Fully functional without backend for development

## License

Proprietary — Aviation Safety Research
