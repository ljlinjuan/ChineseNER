from mosek.fusion import Domain, Expr, Model, ObjectiveSense, Var


def mosek_max_expect_return(n,
                             mu,
                             GT,
                             TE,
                             ind_dist,
                             ind_bound,
                             style_expo,
                             style_bound,
                             idx_wgt,
                             set0_index,
                             wgt_deviation,
                             isin_idx,
                             wgt_in_idx_budget
                            ):
    with Model("Max Expect Return") as M:
        # Defines the variables (holdings). Shortselling is not allowed.
        x = M.variable("x", n, Domain.greaterThan(0.0))  # Portfolio variables

        # Total budget constraint
        M.constraint('budget', Expr.sum(x), Domain.equalsTo(1))

        # Risk neutral with index / or tracking error
        M.constraint('tracking', Expr.vstack(TE, Expr.mul(GT, Expr.sub(x, idx_wgt))), Domain.inQCone())

        M.constraint('weight deviation-', Expr.sub(Expr.sub(x, idx_wgt),
                                                   -wgt_deviation
                                                   ), Domain.greaterThan(0))

        M.constraint('weight deviation', Expr.sub(Expr.sub(x, idx_wgt),
                                                  wgt_deviation
                                                  ), Domain.lessThan(0))
        # # Industry distribution
        M.constraint('industry-', Expr.sub(Expr.mul(ind_dist, Expr.sub(x, idx_wgt)),
                                           -ind_bound
                                           ), Domain.greaterThan(0))

        M.constraint('industry', Expr.sub(Expr.mul(ind_dist, Expr.sub(x, idx_wgt)),
                                          ind_bound
                                          ), Domain.lessThan(0))
        # Style exposure
        M.constraint('style-', Expr.sub(Expr.mul(style_expo,
                                                 Expr.sub(x, idx_wgt)),
                                        -style_bound
                                        ), Domain.greaterThan(0))

        M.constraint('style', Expr.sub(Expr.mul(style_expo,
                                                Expr.sub(x, idx_wgt)),
                                       style_bound
                                       ), Domain.lessThan(0))
        M.constraint('isin_idx_budget', Expr.sub(Expr.sum(Expr.mulElm(isin_idx, x)),
                                                 wgt_in_idx_budget), Domain.greaterThan(0))
        # Bench index members not in universe
        M.constraint('ST', Expr.mulElm(x, set0_index), Domain.equalsTo(0))
        M.objective('obj', ObjectiveSense.Maximize, Expr.dot(mu, Expr.sub(x, idx_wgt)))
        M.solve()

        return x.level()
