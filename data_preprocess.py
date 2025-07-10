from ultralytics.data.split_dota import split_test, split_trainval


if __name__ == '__main__':
    split_trainval(
        data_root="/data/huilin/projects/datasets/DOTAv1",
        save_dir="/data/huilin/projects/datasets/DOTAv1-split1024/",
        crop_size=1024,
        gap=200,
    )
    split_test(
        data_root="/data/huilin/projects/datasets/DOTAv1",
        save_dir="/data/huilin/projects/datasets/DOTAv1-split1024/",
        crop_size=1024,
        gap=200,
    )

    split_trainval(
        data_root="/data/huilin/projects/datasets/DOTAv1",
        save_dir="/data/huilin/projects/datasets/DOTAv1-split640/",
        crop_size=640,
        gap=200,
    )
    split_test(
        data_root="/data/huilin/projects/datasets/DOTAv1",
        save_dir="/data/huilin/projects/datasets/DOTAv1-split640/",
        crop_size=640,
        gap=200,
    )