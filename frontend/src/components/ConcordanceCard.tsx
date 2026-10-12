import React, { useEffect, useState } from 'react';
import { predictConcordance } from '../services/concordanceApi';
import type {
  ConcordanceOutcomeCode,
  ConcordanceResult,
  ImagingOutputInput,
  TabularRiskInput,
} from '../types/concordance';

interface ConcordanceCardProps {
  risk: TabularRiskInput | null;
  imaging: ImagingOutputInput | null;
}

interface OutcomeBadgeConfig {
  badgeBg: string;
  badgeText: string;
  borderColor: string;
  icon: string;
  iconLabel: string;
}

const OUTCOME_CONFIGS: Record<ConcordanceOutcomeCode, OutcomeBadgeConfig> = {
  CONCORDANT_STONE_EVIDENCE: {
    badgeBg: 'bg-[#C97A2B]/15',
    badgeText: 'text-[#B2651A]',
    borderColor: 'border-[#C97A2B]/40',
    icon: '⚠️',
    iconLabel: 'Positive Finding',
  },
  CONCORDANT_LOW_CONCERN: {
    badgeBg: 'bg-[#1F6F5C]/15',
    badgeText: 'text-[#1F6F5C]',
    borderColor: 'border-[#1F6F5C]/40',
    icon: '✓',
    iconLabel: 'Low Concern',
  },
  DISCORDANT: {
    badgeBg: 'bg-[#D97706]/15',
    badgeText: 'text-[#B45309]',
    borderColor: 'border-[#D97706]/40',
    icon: '⇄',
    iconLabel: 'Discordant Streams',
  },
  NON_STONE_FINDING: {
    badgeBg: 'bg-[#DC2626]/15',
    badgeText: 'text-[#B91C1C]',
    borderColor: 'border-[#DC2626]/40',
    icon: '🚨',
    iconLabel: 'Incidental Finding / Clinician Review Required',
  },
  LOW_CONFIDENCE: {
    badgeBg: 'bg-[#4B5563]/15',
    badgeText: 'text-[#374151]',
    borderColor: 'border-[#4B5563]/40',
    icon: '❓',
    iconLabel: 'Low Confidence / Gate Review',
  },
  INSUFFICIENT_DATA: {
    badgeBg: 'bg-[#6B7280]/15',
    badgeText: 'text-[#4B5563]',
    borderColor: 'border-[#6B7280]/40',
    icon: '⏳',
    iconLabel: 'Incomplete Evidence',
  },
};

export const ConcordanceCard: React.FC<ConcordanceCardProps> = ({ risk, imaging }) => {
  const [concordance, setConcordance] = useState<ConcordanceResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasBothStreams = Boolean(
    risk &&
      (risk.probability !== undefined || risk.risk_level !== undefined) &&
      imaging &&
      imaging.class_name
  );

  useEffect(() => {
    if (!hasBothStreams) {
      setConcordance(null);
      setError(null);
      return;
    }

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    predictConcordance({ risk, imaging })
      .then((res) => {
        if (isMounted) {
          setConcordance(res);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to evaluate server-side evidence concordance.'
          );
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [risk, imaging, hasBothStreams]);

  if (!hasBothStreams) {
    return (
      <div className="rounded-xl border border-dashed border-[#DDE3DC] bg-[#F9FAF8] p-5 text-xs text-[#101B16]/70">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-base" aria-hidden="true">
            ⏳
          </span>
          <span className="font-semibold text-sm text-[#101B16]">
            Insufficient Data — Run Both Analyses First
          </span>
        </div>
        <p className="leading-relaxed text-[#101B16]/65">
          Server-side evidence concordance compares independent predictions from urine biomarker
          risk and CT imaging slice detection. Please run both analyses in the input step to
          generate a verified concordance assessment.
        </p>
        <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 border ${
              risk ? 'bg-[#1F6F5C]/10 text-[#1F6F5C] border-[#1F6F5C]/30' : 'bg-gray-100 text-gray-500 border-gray-200'
            }`}
          >
            {risk ? '✓ Biomarker Risk Present' : '○ Biomarker Risk Pending'}
          </span>
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 border ${
              imaging ? 'bg-[#1F6F5C]/10 text-[#1F6F5C] border-[#1F6F5C]/30' : 'bg-gray-100 text-gray-500 border-gray-200'
            }`}
          >
            {imaging ? '✓ CT Slice Present' : '○ CT Slice Pending'}
          </span>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="rounded-xl border border-[#DDE3DC] bg-white p-6 text-center text-xs text-[#101B16]/60">
        <div className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-[#1F6F5C] border-t-transparent mb-2" />
        <p>Evaluating server-side evidence concordance...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50/70 p-5 text-xs text-red-800">
        <p className="font-semibold text-sm mb-1 flex items-center gap-1.5">
          <span>⚠️</span> Concordance Engine Error
        </p>
        <p>{error}</p>
      </div>
    );
  }

  if (!concordance) {
    return null;
  }

  const badgeConfig =
    OUTCOME_CONFIGS[concordance.outcome_code] ?? OUTCOME_CONFIGS.INSUFFICIENT_DATA;

  return (
    <div className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
      {/* Header with Outcome Code and Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#DDE3DC] pb-4 mb-4">
        <div>
          <span className="text-[11px] font-mono tracking-wider uppercase text-[#101B16]/50 block">
            {concordance.rule_fired}
          </span>
          <h3 className="font-serif text-lg font-semibold text-[#101B16] mt-0.5">
            {concordance.outcome_label}
          </h3>
        </div>

        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${badgeConfig.badgeBg} ${badgeConfig.badgeText} ${badgeConfig.borderColor}`}
          role="status"
          aria-label={badgeConfig.iconLabel}
        >
          <span aria-hidden="true" className="text-sm">
            {badgeConfig.icon}
          </span>
          <span>{badgeConfig.iconLabel}</span>
        </div>
      </div>

      {/* Actual Input Probabilities Used (No static placeholders) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        <div className="rounded-lg border border-[#DDE3DC] bg-[#F9FAF8] p-3 text-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[#101B16]/60 font-medium">Urine Biomarker Risk</span>
            <span className="font-semibold text-[#1F6F5C]">
              {risk?.risk_level ?? 'Evaluated'}
            </span>
          </div>
          <p className="text-sm font-semibold text-[#101B16]">
            {risk?.probability !== undefined
              ? `${(risk.probability * 100).toFixed(1)}% probability`
              : 'Probability calculated'}
          </p>
          {risk?.confidence !== undefined && (
            <p className="text-[11px] text-[#101B16]/50 mt-0.5">
              Confidence: {(risk.confidence * 100).toFixed(1)}%
            </p>
          )}
        </div>

        <div className="rounded-lg border border-[#DDE3DC] bg-[#F9FAF8] p-3 text-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[#101B16]/60 font-medium">CT Scan Slice Finding</span>
            <span
              className={`font-semibold ${
                imaging?.class_name === 'Stone' ? 'text-[#C97A2B]' : 'text-[#1F6F5C]'
              }`}
            >
              {imaging?.class_name ?? 'Evaluated'}
            </span>
          </div>
          <p className="text-sm font-semibold text-[#101B16]">
            {imaging?.confidence !== undefined
              ? `${(imaging.confidence * 100).toFixed(1)}% confidence`
              : 'Classification complete'}
          </p>
          {imaging?.status && imaging.status !== 'accepted' && (
            <p className="text-[11px] text-amber-700 font-medium mt-0.5">
              Gate Status: {imaging.status}
            </p>
          )}
        </div>
      </div>

      {/* Rule Explanation */}
      <div className="rounded-lg border border-[#DDE3DC] bg-[#F9FAF8] p-4 text-xs mb-4">
        <p className="font-medium text-[#101B16] mb-1">Concordance Analysis:</p>
        <p className="text-[#101B16]/80 leading-relaxed">{concordance.explanation}</p>
        {concordance.clinician_review_recommended && (
          <div className="mt-2.5 pt-2 border-t border-[#DDE3DC] flex items-center gap-1.5 text-amber-800 font-medium">
            <span aria-hidden="true">⚠️</span>
            <span>Clinician Review Advised: Case warrants secondary review by attending physician.</span>
          </div>
        )}
      </div>

      {/* Regulatory & Multi-Stream Unpaired Dataset Disclaimer */}
      <div className="rounded-lg border border-stone-200 bg-stone-50 p-3 text-[11px] text-stone-600 leading-relaxed">
        <strong className="block text-stone-700 font-semibold mb-0.5">Clinical Disclaimer:</strong>
        {concordance.disclaimer}
      </div>
    </div>
  );
};

export default ConcordanceCard;
