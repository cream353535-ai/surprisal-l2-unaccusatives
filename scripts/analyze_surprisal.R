# analyze_surprisal.R — statistical analyses reported in the paper (Sections 3.5 and 4).
#
# Inputs  (data/):
#   human_rt_trials_pos4-8.csv  trial-level reading times at word positions 4-8
#                               columns: subject, item, condition, position, wpos, rt
#   surprisal_long.csv          output of scripts/compute_surprisal.py
#   stimuli.csv                 item -> pair / animacy / alternation
# Outputs (results/): CSV tables, one per analysis block (see the write_tbl() calls below).
#
# Run from the repository root:   Rscript scripts/analyze_surprisal.R
#
# Analyses
#   1. Trimming (100-3000 ms), log RT, exclusion rates by condition x position
#   2. Omnibus LMM: log RT ~ animacy * alternation * position + (1|subject) + (1|pair), ML, LRTs
#   3. Position-specific effects: by-participant one-sample t tests, per-position LMMs,
#      item-level tests (paired t for animacy over 16 pairs; Welch t for alternation, 8 vs 8)
#   4. Condition effects on surprisal / entropy: lm per model per position (n = 32)
#   5. Human-model correlations: sentence-position means (n = 32 per position; 160 overall;
#      overall with position means removed)
#   6. Trial-level baseline vs. surprisal-augmented LMMs; cue-effect chi-square before/after
#   7. was / being: current + preceding-word surprisal; LRT for animacy * alternation
#   8. Contextual entropy: entropy alone and entropy on top of surprisal

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(lme4)
})
options(dplyr.summarise.inform = FALSE)

.args <- commandArgs(FALSE); .f <- sub("--file=", "", .args[grepl("--file=", .args)])
root <- if (length(.f)) normalizePath(file.path(dirname(.f[1]), ".."), mustWork = FALSE) else "."
if (!dir.exists(file.path(root, "data"))) root <- "."   # fall back to the working directory
dir.create(file.path(root, "results"), showWarnings = FALSE)
write_tbl <- function(x, name) {
  write.csv(x, file.path(root, "results", paste0(name, ".csv")), row.names = FALSE)
  cat("wrote results/", name, ".csv\n", sep = "")
}

POS <- c("Verb", "Det", "NP2", "was", "being")
code_cues <- function(d) {
  d %>% mutate(
    anim = ifelse(animacy == "animate", 0.5, -0.5),          # +0.5 animate
    alt  = ifelse(alternation == "alternating", 0.5, -0.5),  # +0.5 alternating
    position = factor(position, levels = POS))               # treatment coding, Verb = reference
}

# ------------------------------------------------------------------ data
stim <- read.csv(file.path(root, "data", "stimuli.csv"))
sur  <- read.csv(file.path(root, "data", "surprisal_long.csv")) %>%
  filter(wpos >= 4, wpos <= 8) %>% code_cues()
models <- unique(sur$model)

hum <- read.csv(file.path(root, "data", "human_rt_trials_pos4-8.csv")) %>%
  select(subject, item, position, wpos, rt) %>%
  left_join(stim %>% select(item, pair, condition, animacy, alternation), by = "item") %>%
  left_join(sur %>% filter(model == models[1]) %>% select(item, wpos, word, word_length),
            by = c("item", "wpos")) %>%
  code_cues() %>%
  mutate(subject = factor(subject), pair = factor(pair))

# ------------------------------------------------------------------ 1. trimming
hum <- hum %>% mutate(excluded = rt < 100 | rt > 3000)
excl <- hum %>% group_by(condition, position) %>%
  summarise(n = n(), excluded = mean(excluded), above_3000 = mean(rt > 3000)) %>%
  bind_rows(hum %>% summarise(condition = "ALL", position = "ALL", n = n(),
                              excluded = mean(excluded), above_3000 = mean(rt > 3000)))
write_tbl(excl, "01_exclusion_rates")
hum <- hum %>% filter(!excluded) %>% mutate(logrt = log(rt))

rt_means <- hum %>% group_by(subject, condition, position) %>% summarise(rt = mean(rt)) %>%
  group_by(condition, position) %>% summarise(mean_rt = mean(rt), se_rt = sd(rt) / sqrt(n())) %>%
  pivot_wider(names_from = position, values_from = c(mean_rt, se_rt))
write_tbl(rt_means, "02_rt_means_by_condition_position")

# ------------------------------------------------------------------ 2. omnibus LMM with LRTs
m_full <- lmer(logrt ~ anim * alt * position + (1 | subject) + (1 | pair), data = hum, REML = FALSE)
lrt_drop <- function(model, term) {
  reduced <- update(model, as.formula(paste(". ~ . -", term)))
  a <- anova(reduced, model)
  data.frame(term = term, chisq = a$Chisq[2], df = a$Df[2], p = a$`Pr(>Chisq)`[2])
}
omni <- bind_rows(lapply(c("anim", "alt", "anim:alt", "anim:position", "alt:position",
                           "anim:alt:position"), function(t) lrt_drop(m_full, t)))
write_tbl(omni, "03_omnibus_lmm_lrt")

# ------------------------------------------------------------------ 3. position-specific effects
# (a) by-participant contrasts on condition-mean log RT, one-sample t tests
subj_cond <- hum %>% group_by(subject, position, condition) %>% summarise(m = mean(logrt)) %>%
  pivot_wider(names_from = condition, values_from = m) %>%
  mutate(animacy_c  = (AA + AN) / 2 - (IA + IN) / 2,
         altern_c   = (AA + IA) / 2 - (AN + IN) / 2,
         interact_c = (AA - IA) - (AN - IN))
ttest_row <- function(x, lab, pos) {
  x <- x[!is.na(x)]; tt <- t.test(x)
  data.frame(position = pos, effect = lab, b = mean(x), t = unname(tt$statistic),
             df = unname(tt$parameter), p = tt$p.value)
}
by_subj <- bind_rows(lapply(POS, function(p) {
  d <- subj_cond %>% filter(position == p)
  bind_rows(ttest_row(d$animacy_c, "animacy", p), ttest_row(d$altern_c, "alternation", p),
            ttest_row(d$interact_c, "interaction", p))
}))
write_tbl(by_subj, "04a_position_by_participant_ttests")

# (b) per-position LMM, random intercepts for subject and pair, ML
by_pos_lmm <- bind_rows(lapply(POS, function(p) {
  m <- lmer(logrt ~ anim * alt + (1 | subject) + (1 | pair), data = filter(hum, position == p),
            REML = FALSE)
  cf <- summary(m)$coefficients
  data.frame(position = p, effect = c("animacy", "alternation", "interaction"),
             b = cf[c("anim", "alt", "anim:alt"), "Estimate"],
             se = cf[c("anim", "alt", "anim:alt"), "Std. Error"],
             t = cf[c("anim", "alt", "anim:alt"), "t value"])
}))
write_tbl(by_pos_lmm, "04b_position_mixed_models")

# (c) item-level tests on sentence means of log RT (participant-mean per sentence)
sent_mean <- hum %>% group_by(item, pair, condition, animacy, alternation, position) %>%
  summarise(logrt = mean(logrt), .groups = "drop")
item_tests <- bind_rows(lapply(POS, function(p) {
  d <- sent_mean %>% filter(position == p)
  w <- d %>% select(pair, animacy, alternation, logrt) %>%
    pivot_wider(names_from = animacy, values_from = logrt) %>% mutate(diff = animate - inanimate)
  ta <- t.test(w$animate, w$inanimate, paired = TRUE)
  pm <- w %>% mutate(pm = (animate + inanimate) / 2)
  tb <- t.test(pm$pm[pm$alternation == "alternating"], pm$pm[pm$alternation == "non-alternating"])
  tc <- t.test(w$diff[w$alternation == "alternating"], w$diff[w$alternation == "non-alternating"])
  data.frame(position = p,
             effect = c("animacy (paired, 16 pairs)", "alternation (Welch, 8 vs 8)",
                        "interaction (Welch on animacy diff, 8 vs 8)"),
             t = c(ta$statistic, tb$statistic, tc$statistic),
             df = c(ta$parameter, tb$parameter, tc$parameter),
             p = c(ta$p.value, tb$p.value, tc$p.value),
             n_pairs_animate_slower = c(sum(w$diff > 0), NA, NA))
}))
write_tbl(item_tests, "04c_position_item_level_tests")

# leave-one-pair-out check on the direction of the animacy contrast at was / being
loo <- bind_rows(lapply(c("was", "being"), function(p) {
  d <- sent_mean %>% filter(position == p)
  bind_rows(lapply(unique(d$pair), function(pr) {
    dd <- d %>% filter(pair != pr) %>% group_by(animacy) %>% summarise(m = mean(logrt))
    data.frame(position = p, dropped_pair = pr,
               animacy_diff = dd$m[dd$animacy == "animate"] - dd$m[dd$animacy == "inanimate"])
  }))
}))
write_tbl(loo, "04d_leave_one_pair_out_animacy")

# ------------------------------------------------------------------ 4. condition effects on LM measures
lm_cue_effects <- function(dv) {
  bind_rows(lapply(models, function(mk) bind_rows(lapply(POS, function(p) {
    d <- sur %>% filter(model == mk, position == p)
    f <- lm(as.formula(paste(dv, "~ anim * alt")), data = d)
    cf <- summary(f)$coefficients
    data.frame(model = mk, position = p, effect = c("animacy", "alternation", "interaction"),
               b = cf[c("anim", "alt", "anim:alt"), 1], t = cf[c("anim", "alt", "anim:alt"), 3],
               p = cf[c("anim", "alt", "anim:alt"), 4])
  }))))
}
write_tbl(lm_cue_effects("surprisal"), "05a_surprisal_cue_effects_by_model")
write_tbl(lm_cue_effects("entropy"),   "05b_entropy_cue_effects_by_model")

lm_means <- sur %>% group_by(model, condition, position) %>%
  summarise(surprisal = mean(surprisal), entropy = mean(entropy)) %>%
  group_by(condition, position) %>%
  summarise(surprisal = mean(surprisal), entropy = mean(entropy)) %>%
  pivot_wider(names_from = position, values_from = c(surprisal, entropy))
write_tbl(lm_means, "05c_lm_measure_means_6model_average")

# ------------------------------------------------------------------ 5. correlations
sp <- sent_mean %>% left_join(sur %>% select(model, item, position, surprisal, entropy),
                              by = c("item", "position"), relationship = "many-to-many")
cor_row <- function(x, y, lab, mk, p) {
  ct <- cor.test(x, y)
  data.frame(model = mk, measure = lab, position = p, r = unname(ct$estimate),
             n = length(x), p = ct$p.value)
}
cors <- bind_rows(lapply(models, function(mk) {
  d <- sp %>% filter(model == mk)
  dr <- d %>% group_by(position) %>%
    mutate(across(c(logrt, surprisal, entropy), ~ .x - mean(.x), .names = "{.col}_c")) %>% ungroup()
  bind_rows(
    cor_row(d$logrt, d$surprisal, "surprisal", mk, "overall (160)"),
    cor_row(dr$logrt_c, dr$surprisal_c, "surprisal", mk, "overall, position means removed"),
    cor_row(d$logrt, d$entropy, "entropy", mk, "overall (160)"),
    cor_row(dr$logrt_c, dr$entropy_c, "entropy", mk, "overall, position means removed"),
    bind_rows(lapply(POS, function(p) {
      dd <- d %>% filter(position == p)
      bind_rows(cor_row(dd$logrt, dd$surprisal, "surprisal", mk, p),
                cor_row(dd$logrt, dd$entropy, "entropy", mk, p))
    })))
}))
write_tbl(cors, "06_correlations_rt_vs_lm_measures")

# entropy before `was` vs surprisal at `was`, by model (n = 32)
ent_vs_surp <- bind_rows(lapply(models, function(mk) {
  d <- sur %>% filter(model == mk, position == "was")
  data.frame(model = mk, r = cor(d$entropy, d$surprisal))
}))
write_tbl(ent_vs_surp, "06b_entropy_before_was_vs_surprisal_at_was")

# ------------------------------------------------------------------ 6. trial-level baseline vs augmented
trial <- hum %>% left_join(sur %>% select(model, item, wpos, surprisal, entropy),
                           by = c("item", "wpos"), relationship = "many-to-many")
cue_terms <- "anim + alt + anim:alt + anim:position + alt:position + anim:alt:position"
f_base <- as.formula(paste("logrt ~ anim * alt * position + word_length + (1|subject) + (1|pair)"))
f_nocue <- as.formula("logrt ~ position + word_length + (1|subject) + (1|pair)")

augmented <- bind_rows(lapply(models, function(mk) {
  d <- trial %>% filter(model == mk) %>%
    mutate(z_surp = as.numeric(scale(surprisal)), z_ent = as.numeric(scale(entropy)))
  m0 <- lmer(f_base, data = d, REML = FALSE)
  m1 <- update(m0, . ~ . + z_surp)
  m0_nocue <- lmer(f_nocue, data = d, REML = FALSE)
  m1_nocue <- update(m0_nocue, . ~ . + z_surp)
  a_surp <- anova(m0, m1)
  cue_before <- anova(m0_nocue, m0); cue_after <- anova(m1_nocue, m1)
  # entropy
  mE  <- update(m0, . ~ . + z_ent);  a_ent  <- anova(m0, mE)
  mSE <- update(m1, . ~ . + z_ent);  a_sent <- anova(m1, mSE)
  cf <- summary(m1)$coefficients["z_surp", ]
  data.frame(model = mk,
             surprisal_b = cf["Estimate"], surprisal_se = cf["Std. Error"], surprisal_t = cf["t value"],
             lrt_chisq = a_surp$Chisq[2], lrt_p = a_surp$`Pr(>Chisq)`[2],
             delta_aic = AIC(m1) - AIC(m0),
             cue_chisq_before = cue_before$Chisq[2], cue_df = cue_before$Df[2],
             cue_chisq_after = cue_after$Chisq[2],
             cue_reduction_pct = 100 * (1 - cue_after$Chisq[2] / cue_before$Chisq[2]),
             entropy_alone_chisq = a_ent$Chisq[2], entropy_alone_p = a_ent$`Pr(>Chisq)`[2],
             entropy_over_surprisal_chisq = a_sent$Chisq[2],
             entropy_over_surprisal_p = a_sent$`Pr(>Chisq)`[2])
}))
write_tbl(augmented, "07_trial_level_baseline_vs_augmented")

# ------------------------------------------------------------------ 7. was / being with current + preceding surprisal
prev <- sur %>% mutate(wpos = wpos + 1) %>% select(model, item, wpos, surp_prev = surprisal)
wb <- bind_rows(lapply(c("was", "being"), function(p) {
  d0 <- hum %>% filter(position == p)
  m_noS <- lmer(logrt ~ anim * alt + (1 | subject) + (1 | pair), data = d0, REML = FALSE)
  cf0 <- summary(m_noS)$coefficients
  row0 <- data.frame(position = p, model = "none (no surprisal)",
                     anim_b = cf0["anim", 1], anim_t = cf0["anim", 3],
                     alt_b = cf0["alt", 1], alt_t = cf0["alt", 3],
                     int_b = cf0["anim:alt", 1], int_t = cf0["anim:alt", 3],
                     lrt_cues_chisq = NA, lrt_cues_df = NA, lrt_cues_p = NA,
                     lrt_anim_chisq = NA, lrt_anim_p = NA)
  rows <- bind_rows(lapply(models, function(mk) {
    d <- d0 %>%
      left_join(sur %>% filter(model == mk) %>% select(item, wpos, surp_cur = surprisal),
                by = c("item", "wpos")) %>%
      left_join(prev %>% filter(model == mk) %>% select(item, wpos, surp_prev), by = c("item", "wpos")) %>%
      mutate(z_cur = as.numeric(scale(surp_cur)), z_prev = as.numeric(scale(surp_prev)))
    mS  <- lmer(logrt ~ z_cur + z_prev + (1 | subject) + (1 | pair), data = d, REML = FALSE)
    mSC <- update(mS, . ~ . + anim * alt)
    mSC_noA <- update(mSC, . ~ . - anim)
    a <- anova(mS, mSC); aA <- anova(mSC_noA, mSC)
    cf <- summary(mSC)$coefficients
    data.frame(position = p, model = mk,
               anim_b = cf["anim", 1], anim_t = cf["anim", 3],
               alt_b = cf["alt", 1], alt_t = cf["alt", 3],
               int_b = cf["anim:alt", 1], int_t = cf["anim:alt", 3],
               lrt_cues_chisq = a$Chisq[2], lrt_cues_df = a$Df[2], lrt_cues_p = a$`Pr(>Chisq)`[2],
               lrt_anim_chisq = aA$Chisq[2], lrt_anim_p = aA$`Pr(>Chisq)`[2])
  }))
  bind_rows(row0, rows)
}))
write_tbl(wb, "08_was_being_current_plus_preceding_surprisal")

cat("\nDone. Tables are in results/.\n")
