import React, { useState } from 'react';
import axios from 'axios';
import ClauseTree from './ClauseTree';

interface Clause {
  number: string;
  title: string;
  content: string;
  level?: number;
  children?: Clause[];
}

interface ParseResponse {
  clauses: Clause[];
  metadata: {
    total_pages: number;
    processing_date?: string;
    filename?: string;
  };
  total_pages: number;
  processing_time: number;
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const ContractUploader: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [clauses, setClauses] = useState<Clause[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ParseResponse | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setClauses([]);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<ParseResponse>(
        `${API_URL}/api/parse-contract`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 300000, // 5 minutes timeout
        }
      );

      setClauses(response.data.clauses);
      setResult(response.data);
    } catch (err: any) {
      console.error('Error parsing contract:', err);
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to parse contract. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-8">
      {/* Upload Interface */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Upload Contract</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select PDF File
            </label>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100
                cursor-pointer"
              disabled={loading}
            />
          </div>

          {file && (
            <div className="text-sm text-gray-600">
              Selected: <span className="font-medium">{file.name}</span>
              {' '}
              ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full bg-blue-600 text-white px-6 py-3 rounded-lg
              font-medium hover:bg-blue-700 disabled:bg-gray-400
              disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing Contract...
              </span>
            ) : (
              'Parse Contract'
            )}
          </button>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Results Summary */}
      {result && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Processing Results</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-600">Total Pages</div>
              <div className="text-2xl font-bold text-gray-900">
                {result.total_pages}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Clauses Found</div>
              <div className="text-2xl font-bold text-gray-900">
                {result.clauses.length}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Processing Time</div>
              <div className="text-2xl font-bold text-gray-900">
                {result.processing_time}s
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Clause Tree Visualization */}
      {clauses.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Extracted Clauses</h2>
          <ClauseTree clauses={clauses} />
        </div>
      )}
    </div>
  );
};

export default ContractUploader;

