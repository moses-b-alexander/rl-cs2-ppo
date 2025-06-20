


def show_column_info(final: DataFrame) -> None:

    print()
    for c in list(final.columns):
        for k0 in ["node", "downsample", "graph"]:
            for k1 in ["situation", "round", "game", "match"]:
                if c in list(cols[k0][k1].keys()):
                    if k0 == "node":  s0 = "agent"
                    else:  s0 = "global"
                    s1 = k1
        print(c, " : ", final[c].dtype, " ; ", s0, " , ", s1, " . ")
        ll = list(factorize(final[c])[1])
        if "float" in str(final[c].dtype) or "int" in str(final[c].dtype):
            print(
                "num || ", len(ll), "\n",
                final[c].min(), final[c].mean(), final[c].max(),
                "\nmode || ", final[c].mode().iloc[0]
            )
        else:
            print(
                "cat || ", len(ll), "\n", final[c].value_counts(),
                "\nmode || ", final[c].mode().iloc[0]
            )
        print()
    print()

    return None


role    = "arn:aws:iam::123456789012:role/SageMakerRole"
s3_uri  = "s3://my-bucket/ct_rounds.npz"

estimator = RLEstimator(
    entry_point    = "train_local.py",   # uses exactly the same script
    source_dir     = ".",                # must include cs2_env.py & train_local.py
    role           = role,
    instance_count = 1,
    instance_type  = "ml.c5.4xlarge",
    framework      = "ray",
    framework_version="1.13.0",          # or your supported version
    py_version     = "py3",
    hyperparameters={
        "data-path": s3_uri,
        "stop-iters": 50,
    }
)

# Tell SageMaker where to find the offline data
estimator.fit({"training": s3_uri})


