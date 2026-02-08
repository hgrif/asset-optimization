- Modern pandas
- Polars instead of pandas
- Explain Weibull
- Intervention in the simulator
```
        n_failures = failures_mask.sum()
        if n_failures > 0:
            # Failure costs
            costs['failure_direct'] = n_failures * DEFAULT_FAILURE_DIRECT_COST
            costs['failure_consequence'] = n_failures * DEFAULT_FAILURE_CONSEQUENCE_COST

            # Apply intervention based on failure_response config
            if self.config.failure_response == 'replace':
                intervention = self.interventions.get('replace', REPLACE)
                state.loc[failures_mask, 'age'] = state.loc[failures_mask, 'age'].apply(
                    intervention.apply_age_effect
                )
                costs['intervention'] = n_failures * intervention.cost

            elif self.config.failure_response == 'repair':
                intervention = self.interventions.get('repair', REPAIR)
                state.loc[failures_mask, 'age'] = state.loc[failures_mask, 'age'].apply(
                    intervention.apply_age_effect
                )
                costs['intervention'] = n_failures * intervention.cost
```

- can we use pyomo?
- use end2end test
- switch to callable interface?
