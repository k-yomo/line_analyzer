package functions

import (
	"context"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/bigquery"
	"cloud.google.com/go/storage"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rekognition"
	"github.com/pkg/errors"
	"github.com/rs/xid"
)

const (
	fastLaneDataset          = "fast_lane"
	lineObservationTable     = "line_observation"
	waitingCustomerMetaTable = "waiting_customer_meta"
)

type gcsEvent struct {
	Bucket      string `json:"bucket"`
	ObjectName  string `json:"name"`
	ContentType string `json:"contentType"`
}

type lineObservation struct {
	ID     string `bigquery:"id"`
	ShopID string `bigquery:"shop_id"`
	// WaitingPeopleNum is more accurate than number of DetectedHumans since DetectedHumans is detected by their face
	WaitingPeopleNum int       `bigquery:"waiting_people_num"`
	ObservedAt       time.Time `bigquery:"observed_at"`
	CreatedAt        time.Time `bigquery:"created_at"`
}

type waitingCustomerMeta struct {
	LineObservationID string  `bigquery:"line_observation_id"`
	Gender            string  `bigquery:"gender"`
	GenderConfidence  float64 `bigquery:"gender_confidence"`
	LowestAge         int64   `bigquery:"lowest_age"`
	HighestAge        int64   `bigquery:"highest_age"`
	Confidence        float64 `bigquery:"confidence"`
}

// AnalyzeLineImage analyzes the image of lineObservation at shop.
func AnalyzeLineImage(ctx context.Context, e gcsEvent) error {
	now := time.Now()

	gcsClient, err := storage.NewClient(ctx)
	if err != nil {
		return errors.Wrap(err, "init gcs client")
	}
	bqClient, err := bigquery.NewClient(ctx, mustEnv("GCP_PROJECT_ID"))
	if err != nil {
		return errors.Wrap(err, "init bigquery client failed")
	}

	obj := gcsClient.Bucket(e.Bucket).Object(e.ObjectName)
	lineObservationID := xid.New().String()
	shopID, observedAt, err := getMetaFromObjName(obj.ObjectName())
	if err != nil {
		return errors.Wrapf(err, "get lineObservation meta from %s", obj.ObjectName())
	}

	reader, err := obj.NewReader(ctx)
	if err != nil {
		return errors.Wrapf(err, "new %s reader failed", obj.ObjectName())
	}
	waitingCustomerNum, customers, err := detectWaitingCustomersFromImg(ctx, lineObservationID, reader)
	if err != nil {
		return errors.Wrapf(err, "analyze %s image failed", obj.ObjectName())
	}

	lo := &lineObservation{
		ID:               lineObservationID,
		ShopID:           shopID,
		WaitingPeopleNum: waitingCustomerNum,
		ObservedAt:       observedAt,
		CreatedAt:        now,
	}

	ds := bqClient.Dataset(fastLaneDataset)
	if err := ds.Table(lineObservationTable).Inserter().Put(ctx, []*lineObservation{lo}); err != nil {
		return errors.Wrap(err, "put analyzed lineObservation data")
	}
	if err := ds.Table(waitingCustomerMetaTable).Inserter().Put(ctx, customers); err != nil {
		return errors.Wrap(err, "load analyzed lineObservation data to bq faile")
	}
	return nil
}

func getMetaFromObjName(objName string) (shopID string, observedAt time.Time, err error) {
	name := filepath.Base(objName[:len(objName)-len(filepath.Ext(objName))])
	objMetas := strings.Split(name, "_")
	shopID = objMetas[0]
	unixTime, err := strconv.ParseInt(objMetas[1], 10, 64)
	if err != nil {
		return shopID, time.Time{}, err
	}
	return shopID, time.Unix(unixTime, 0).Local(), err
}

func detectWaitingCustomersFromImg(ctx context.Context, lineObservationID string, imgReader io.Reader) (waitingPeopleNum int, humans []*waitingCustomerMeta, err error) {
	sess, err := session.NewSession()
	if err != nil {
		return 0, nil, errors.Wrap(err, "new aws session")
	}
	svc := rekognition.New(sess, aws.NewConfig().WithRegion("ap-northeast-1"))
	bytes, err := ioutil.ReadAll(imgReader)
	if err != nil {
		return 0, nil, errors.Wrap(err, "read img bytes from reader")
	}
	detectLabelOutput, err := svc.DetectLabelsWithContext(ctx, &rekognition.DetectLabelsInput{Image: &rekognition.Image{Bytes: bytes}})
	if err != nil {
		return 0, nil, errors.Wrap(err, "detect labels")
	}
	for _, label := range detectLabelOutput.Labels {
		if *label.Confidence < 0.5 {
			continue
		}
		if *label.Name == "Person" {
			waitingPeopleNum = len(label.Instances)
		}
	}

	detectFacesOutput, err := svc.DetectFacesWithContext(ctx, &rekognition.DetectFacesInput{
		Attributes: []*string{aws.String("ALL")},
		Image:      &rekognition.Image{Bytes: bytes},
	})
	if err != nil {
		return 0, nil, errors.Wrap(err, "detect faces")
	}

	var customers []*waitingCustomerMeta
	for _, faceDetail := range detectFacesOutput.FaceDetails {
		cus := &waitingCustomerMeta{
			LineObservationID: lineObservationID,
			Gender:            *faceDetail.Gender.Value,
			GenderConfidence:  *faceDetail.Gender.Confidence,
			LowestAge:         *faceDetail.AgeRange.Low,
			HighestAge:        *faceDetail.AgeRange.High,
			Confidence:        *faceDetail.Confidence,
		}
		customers = append(customers, cus)
	}
	return waitingPeopleNum, customers, nil
}

func mustEnv(key string) string {
	env := os.Getenv(key)
	if env == "" {
		log.Fatal(errors.Errorf("env variable with key=%s is not found", key))
	}
	return env
}
