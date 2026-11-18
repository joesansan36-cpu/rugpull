// Existing app code uses localhost:3000 for its same-origin Flask API. Rewrite
// those URLs when hosted so browser requests go to the deployed Render service.
const localApi = 'http://localhost:3000';
const nativeFetch = window.fetch.bind(window);

window.fetch = (input, init) => {
    if (typeof input === 'string' && input.startsWith(localApi)) {
        input = `${window.location.origin}${input.slice(localApi.length)}`;
    } else if (input instanceof URL && input.origin === localApi) {
        input = new URL(input.pathname + input.search + input.hash, window.location.origin);
    }
    return nativeFetch(input, init);
};
