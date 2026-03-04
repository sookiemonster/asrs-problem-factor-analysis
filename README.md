# asrs-problem-factor-analysis

## Notes To Consider

Data Fetching:

- CSV export does not include Narrative 3 or more. Consider ACN 1877264, which has 3 narratives, but the third is non-existent.
- Sometimes, narrative 1 exports as NaN. We fix these by fetching the ACNs individually and including them in `0_keep_first.csv` (since we duduplicate by using keep-first, and glob reads files in alphabetical/numeric order)
