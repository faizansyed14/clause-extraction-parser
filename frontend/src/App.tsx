import React from 'react';
import ContractUploader from './components/ContractUploader';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-3xl font-bold text-gray-900">
            Legal Contract Parser
          </h1>
          <p className="text-gray-600 mt-1">
            Advanced AI-powered contract clause extraction system
          </p>
        </div>
      </header>
      <main>
        <ContractUploader />
      </main>
    </div>
  );
}

export default App;

