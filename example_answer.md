# Question: "How to exclude the minimum dependency rule in gradleLint plugin?"

# Answer
To exclude the minimum dependency rule in gradleLint plugin, you can use the gradle property gradleLint.excludedRules or the extension property of the same name. This can be done by adding the following code to your build.gradle file:

```
gradleLint {
   excludedRules = ['minimum-dependency-version']
}
```

This will exclude the minimum dependency rule from running when gradleLint is applied. You can also add multiple rules to the excludedRules list by separating them with a comma.
