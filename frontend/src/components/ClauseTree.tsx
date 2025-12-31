import React, { useState } from 'react';

interface Clause {
  number: string;
  title: string;
  content: string;
  type: string;
  confidence: number;
  level?: number;
  children?: Clause[];
}

interface ClauseTreeProps {
  clauses: Clause[];
}

const ClauseTree: React.FC<ClauseTreeProps> = ({ clauses }) => {
  return (
    <div className="space-y-3">
      {clauses.map((clause, index) => (
        <ClauseCard key={`${clause.number}-${index}`} clause={clause} />
      ))}
    </div>
  );
};

const ClauseCard: React.FC<{ clause: Clause }> = ({ clause }) => {
  const [expanded, setExpanded] = useState(false);

  const ChevronIcon: React.FC<{ expanded: boolean }> = ({ expanded }) => (
    <svg
      className={`w-5 h-5 text-gray-500 transition-transform ${
        expanded ? 'transform rotate-90' : ''
      }`}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 5l7 7-7 7"
      />
    </svg>
  );

  const getTypeColor = (type: string) => {
    const colors: { [key: string]: string } = {
      parties: 'bg-blue-100 text-blue-800',
      effective_date: 'bg-green-100 text-green-800',
      expiration_date: 'bg-red-100 text-red-800',
      governing_law: 'bg-purple-100 text-purple-800',
      confidentiality: 'bg-yellow-100 text-yellow-800',
      indemnity: 'bg-orange-100 text-orange-800',
      liability_cap: 'bg-pink-100 text-pink-800',
      termination: 'bg-red-100 text-red-800',
      ip_ownership: 'bg-indigo-100 text-indigo-800',
      definitions: 'bg-gray-100 text-gray-800',
      general: 'bg-gray-100 text-gray-800',
    };
    return colors[type] || colors.general;
  };

  return (
    <div className="border border-gray-200 rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow">
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50 rounded-t-lg"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3 flex-wrap">
          <span className="font-mono text-blue-600 font-semibold text-sm">
            {clause.number}
          </span>
          <span className="font-semibold text-gray-900">{clause.title}</span>
          <span
            className={`text-xs px-2 py-1 rounded-full font-medium ${getTypeColor(
              clause.type
            )}`}
          >
            {clause.type.replace(/_/g, ' ')}
          </span>
          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
            {(clause.confidence * 100).toFixed(1)}% confidence
          </span>
        </div>
        <div className="flex items-center gap-2">
          {clause.children && clause.children.length > 0 && (
            <span className="text-xs text-gray-500">
              {clause.children.length} sub-clause{clause.children.length !== 1 ? 's' : ''}
            </span>
          )}
          <ChevronIcon expanded={expanded} />
        </div>
      </div>

      {expanded && (
        <div className="px-4 pb-4 pt-2 border-t border-gray-100">
          <div className="mt-3">
            <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
              {clause.content}
            </p>
          </div>

          {clause.children && clause.children.length > 0 && (
            <div className="mt-4 ml-4 border-l-2 border-blue-200 pl-4 space-y-2">
              <div className="text-sm font-medium text-gray-600 mb-2">
                Sub-clauses:
              </div>
              {clause.children.map((child, index) => (
                <ClauseCard key={`${child.number}-${index}`} clause={child} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ClauseTree;

