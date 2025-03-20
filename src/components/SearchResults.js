const SearchResults = ({ results }) => {
  return (
    <div className="mt-5">
      <h2 className="text-xl font-bold">Search Results:</h2>
      <ul className="mt-3">
        {results.length === 0 ? (
          <li>No results found.</li>
        ) : (
          results.map((result, index) => <li key={index} className="p-2 border">{result}</li>)
        )}
      </ul>
    </div>
  );
};

export default SearchResults;

